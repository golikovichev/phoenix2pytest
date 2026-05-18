"""MCP client for the @arizeai/phoenix-mcp server.

Spawns the official Arize Phoenix MCP server as an `npx` subprocess and
communicates with it via JSON-RPC over stdio. Exposes Python-friendly
methods for the tools phoenix2pytest needs.

Why subprocess + raw JSON-RPC instead of a Python MCP SDK: keeps the
runtime dependency surface small (stdlib only on the Python side), and
makes the integration transparent to read in the repository. The Phoenix
MCP server is shipped as an npm package and `npx` handles install on
first run.

Hackathon context: rule 7.B requires integration with a Partner Entity's
MCP server. This module is the bridge between the rest of the agent
(written in Python) and the Phoenix observability platform exposed
through the Model Context Protocol.

Environment variables (loaded by callers; this module reads what is
passed in `PhoenixMCPConfig`):

- `PHOENIX_HOST`: base URL of the Phoenix instance (e.g.
  `https://app.phoenix.arize.com`)
- `PHOENIX_API_KEY`: API key for the Phoenix workspace
- `PHOENIX_PROJECT`: optional default project for project-scoped calls
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class PhoenixMCPConfig:
    """Configuration for the Phoenix MCP server subprocess."""

    base_url: str
    api_key: str
    project: str | None = None

    @classmethod
    def from_env(cls) -> PhoenixMCPConfig:
        """Build config from PHOENIX_HOST / PHOENIX_API_KEY / PHOENIX_PROJECT."""
        try:
            return cls(
                base_url=os.environ["PHOENIX_HOST"],
                api_key=os.environ["PHOENIX_API_KEY"],
                project=os.environ.get("PHOENIX_PROJECT"),
            )
        except KeyError as missing:
            raise RuntimeError(
                f"Phoenix MCP config requires env var {missing}. "
                "Set PHOENIX_HOST and PHOENIX_API_KEY (and optional PHOENIX_PROJECT)."
            ) from None


class PhoenixMCPError(RuntimeError):
    """Raised when an MCP tool call returns an error response."""


class PhoenixMCPClient:
    """Synchronous JSON-RPC client for the @arizeai/phoenix-mcp server.

    Use as a context manager:

        with PhoenixMCPClient(config) as mcp:
            projects = mcp.list_projects()
            traces = mcp.list_traces(project_id=projects[0]["id"])

    Internally spawns `npx @arizeai/phoenix-mcp@latest` and tears it down
    on context exit.
    """

    CLIENT_PROTOCOL_VERSION = "2024-11-05"
    SUPPORTED_PROTOCOL_VERSIONS = frozenset({"2024-11-05", "2025-03-26", "2025-06-18"})

    def __init__(self, config: PhoenixMCPConfig, package_version: str = "latest"):
        self._config = config
        self._package_version = package_version
        self._proc: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_buffer: list[str] = []
        self._request_id = 0
        self.server_protocol_version: str | None = None

    # --- lifecycle -----------------------------------------------------

    def __enter__(self) -> PhoenixMCPClient:
        # Phoenix MCP server reads PHOENIX_HOST and PHOENIX_API_KEY from
        # the environment when not passed as CLI flags. Passing the API
        # key through the environment instead of argv keeps it out of
        # `ps`/process-listing output, which is visible to other local
        # processes on shared machines.
        cmd: list[str] = [
            "npx",
            "-y",
            f"@arizeai/phoenix-mcp@{self._package_version}",
        ]
        child_env = dict(os.environ)
        child_env["PHOENIX_HOST"] = self._config.base_url
        child_env["PHOENIX_API_KEY"] = self._config.api_key
        if self._config.project is not None:
            child_env["PHOENIX_PROJECT"] = self._config.project
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            shell=False,
            env=child_env,
        )
        self._start_stderr_reader()
        # If the handshake fails we still own the subprocess; tear it down
        # explicitly because __exit__ is only called when __enter__ returns
        # successfully.
        try:
            self._initialize()
        except BaseException:
            self._teardown()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._teardown()

    def _teardown(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                with contextlib.suppress(BrokenPipeError, OSError):
                    self._proc.stdin.close()
            with contextlib.suppress(ProcessLookupError, OSError):
                self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError, OSError):
                    self._proc.kill()
                    self._proc.wait(timeout=2)
        finally:
            self._proc = None

    # --- transport -----------------------------------------------------

    def _start_stderr_reader(self) -> None:
        assert self._proc is not None
        assert self._proc.stderr is not None

        def reader() -> None:
            assert self._proc is not None and self._proc.stderr is not None
            for line in self._proc.stderr:
                self._stderr_buffer.append(line.rstrip())

        self._stderr_thread = threading.Thread(target=reader, daemon=True)
        self._stderr_thread.start()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, payload: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _scrub_secrets(self, text: str) -> str:
        """Remove the API key substring from log output before raising."""
        if not text:
            return text
        key = self._config.api_key
        return text.replace(key, "[REDACTED]") if key else text

    def _read_one(self) -> dict:
        """Read one JSON-RPC frame from the server. Raise if stdout closed."""
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            stderr_tail = self._scrub_secrets("\n".join(self._stderr_buffer[-20:]))
            raise PhoenixMCPError(
                f"Phoenix MCP server closed stdout without responding. "
                f"Stderr tail: {stderr_tail!r}"
            )
        return json.loads(line)

    def _recv_response(self, request_id: int) -> dict:
        """Read frames until we find the response matching `request_id`.

        MCP servers may emit server-initiated notifications (`notifications/
        message`, `notifications/progress`) between requests and their
        responses. Those carry a `method` field and no `id` for the response
        shape; ignore them and keep reading. Also tolerates server-initiated
        requests (with `id` + `method`) by silently dropping them; we do not
        implement reverse handlers in this client.
        """
        while True:
            frame = self._read_one()
            # Server-initiated notification: no id, has method.
            if "method" in frame and "id" not in frame:
                continue
            # Server-initiated request: has id and method. Drop silently;
            # this client does not advertise capabilities that should trigger
            # server callbacks, but be defensive.
            if "method" in frame and "id" in frame:
                continue
            # Response: must have matching id.
            if frame.get("id") == request_id:
                return frame
            # Response for a different request: should not happen for a
            # serial client, but tolerate by dropping.
            continue

    def _rpc(self, method: str, params: dict | None = None) -> Any:
        request_id = self._next_id()
        payload: dict = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)
        resp = self._recv_response(request_id)
        if "error" in resp:
            scrubbed = self._scrub_secrets(json.dumps(resp["error"]))
            raise PhoenixMCPError(f"{method} failed: {scrubbed}")
        return resp.get("result", {})

    def _notify(self, method: str, params: dict | None = None) -> None:
        payload: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    # --- MCP handshake -------------------------------------------------

    def _initialize(self) -> None:
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": self.CLIENT_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "phoenix2pytest", "version": "0.0.1"},
            },
        )
        server_version = result.get("protocolVersion") if isinstance(result, dict) else None
        if isinstance(server_version, str):
            self.server_protocol_version = server_version
            if server_version not in self.SUPPORTED_PROTOCOL_VERSIONS:
                # Server speaks a version we have not validated against. Log
                # via stderr buffer (visible if subsequent calls fail) rather
                # than raise; many newer protocol revisions stay backward
                # compatible for the small surface we use (initialize +
                # tools/list + tools/call).
                self._stderr_buffer.append(
                    f"[phoenix2pytest] warning: server protocolVersion "
                    f"{server_version!r} not in supported set "
                    f"{sorted(self.SUPPORTED_PROTOCOL_VERSIONS)}"
                )
        self._notify("notifications/initialized")

    # --- tool discovery ------------------------------------------------

    def list_tools(self) -> list[dict]:
        """Return the raw tools/list response. Useful for debugging."""
        result = self._rpc("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        """Invoke a named MCP tool and return its raw result payload."""
        return self._rpc(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )

    # --- typed convenience wrappers -----------------------------------
    # The exact argument names below mirror the @arizeai/phoenix-mcp
    # README tool coverage list. They are intentionally thin pass-through
    # wrappers: anything not covered here can still be reached through
    # `call_tool` directly.

    def list_projects(self) -> Any:
        return self.call_tool("list-projects")

    def list_traces(self, project_id: str | None = None, limit: int = 20) -> Any:
        args: dict[str, Any] = {"limit": limit}
        if project_id is not None:
            args["projectId"] = project_id
        elif self._config.project is not None:
            args["projectId"] = self._config.project
        return self.call_tool("list-traces", args)

    def get_trace(self, trace_id: str) -> Any:
        return self.call_tool("get-trace", {"traceId": trace_id})

    def get_spans(self, trace_id: str) -> Any:
        return self.call_tool("get-spans", {"traceId": trace_id})

    def get_span_annotations(self, span_id: str) -> Any:
        return self.call_tool("get-span-annotations", {"spanId": span_id})
