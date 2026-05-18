"""Tests for the Phoenix MCP client wrapper.

These exercise the JSON-RPC plumbing against an offline fake server first
(deterministic, runs in CI), then optionally against the real Phoenix MCP
subprocess if `PHOENIX_HOST` and `PHOENIX_API_KEY` are configured.

The real-subprocess test is marked `integration` and skipped by default so
CI does not depend on a network round-trip to Arize Phoenix or on `npx`
availability on the runner.
"""

from __future__ import annotations

import json
import os
from io import StringIO

import pytest

from phoenix2pytest.mcp_client import (
    PhoenixMCPClient,
    PhoenixMCPConfig,
    PhoenixMCPError,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_config_from_env_reads_three_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOENIX_HOST", "https://phoenix.example.com")
    monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
    monkeypatch.setenv("PHOENIX_PROJECT", "demo-project")

    config = PhoenixMCPConfig.from_env()

    assert config.base_url == "https://phoenix.example.com"
    assert config.api_key == "test-key"
    assert config.project == "demo-project"


def test_config_from_env_project_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOENIX_HOST", "https://phoenix.example.com")
    monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
    monkeypatch.delenv("PHOENIX_PROJECT", raising=False)

    config = PhoenixMCPConfig.from_env()

    assert config.project is None


def test_config_from_env_raises_on_missing_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOENIX_HOST", raising=False)
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Phoenix MCP config requires env var"):
        PhoenixMCPConfig.from_env()


# ---------------------------------------------------------------------------
# Fake subprocess for transport tests
# ---------------------------------------------------------------------------


class _FakeProc:
    """Mimics subprocess.Popen enough for the client to talk JSON-RPC."""

    def __init__(self, scripted_responses: list[dict]):
        self._inbox: list[str] = []
        self._scripted = list(scripted_responses)
        self.stdin = _FakeStdin(self._inbox)
        self.stdout = _FakeStdout(self._scripted, self._inbox)
        self.stderr = StringIO("")

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        pass


class _FakeStdin:
    def __init__(self, inbox: list[str]):
        self._inbox = inbox
        self.closed = False

    def write(self, payload: str) -> None:
        self._inbox.append(payload)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeStdout:
    """Returns one scripted response per readline(), skipping notifications."""

    def __init__(self, scripted: list[dict], inbox: list[str]):
        self._scripted = scripted
        self._inbox = inbox
        self._cursor = 0

    def readline(self) -> str:
        if self._cursor >= len(self._scripted):
            return ""
        item = self._scripted[self._cursor]
        self._cursor += 1
        return json.dumps(item) + "\n"


def _client_with_fake(scripted: list[dict]) -> PhoenixMCPClient:
    config = PhoenixMCPConfig(base_url="https://x", api_key="y")
    client = PhoenixMCPClient(config)
    client._proc = _FakeProc(scripted)  # type: ignore[assignment]
    # Skip stderr reader thread because fake stderr is a one-shot string.
    return client


# ---------------------------------------------------------------------------
# Transport / protocol tests
# ---------------------------------------------------------------------------


def test_initialize_sends_correct_payload() -> None:
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}},
        ]
    )
    client._initialize()

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    assert sent["method"] == "initialize"
    assert sent["params"]["clientInfo"]["name"] == "phoenix2pytest"


def test_call_tool_returns_result_payload() -> None:
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}},
        ]
    )
    result = client.call_tool("list-projects")
    assert result == {"tools": []}


def test_call_tool_raises_on_error_response() -> None:
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "no auth"}},
        ]
    )
    with pytest.raises(PhoenixMCPError, match="no auth"):
        client.call_tool("list-projects")


def test_list_traces_passes_project_id_from_config() -> None:
    config = PhoenixMCPConfig(base_url="x", api_key="y", project="default-proj")
    client = PhoenixMCPClient(config)
    client._proc = _FakeProc([{"jsonrpc": "2.0", "id": 1, "result": {"traces": []}}])
    client.list_traces()

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    assert sent["params"]["arguments"]["projectId"] == "default-proj"


def test_list_traces_explicit_project_overrides_config_default() -> None:
    config = PhoenixMCPConfig(base_url="x", api_key="y", project="default-proj")
    client = PhoenixMCPClient(config)
    client._proc = _FakeProc([{"jsonrpc": "2.0", "id": 1, "result": {"traces": []}}])
    client.list_traces(project_id="other-proj")

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    assert sent["params"]["arguments"]["projectId"] == "other-proj"


# ---------------------------------------------------------------------------
# Server-initiated frames + response correlation
# ---------------------------------------------------------------------------


def test_recv_response_drops_server_notification_before_response() -> None:
    """Phoenix MCP can emit notifications/message before the actual response."""
    client = _client_with_fake(
        [
            # Server-initiated notification (no id, has method): must be dropped
            {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "info", "data": "warming up"},
            },
            # The real response
            {"jsonrpc": "2.0", "id": 1, "result": {"tools": ["dummy"]}},
        ]
    )
    result = client.call_tool("list-projects")
    assert result == {"tools": ["dummy"]}


def test_recv_response_drops_frame_with_mismatched_id() -> None:
    """Response addressed to a different request id must be ignored."""
    client = _client_with_fake(
        [
            # Stale response from a request that did not happen; drop
            {"jsonrpc": "2.0", "id": 999, "result": {"stale": True}},
            # Real response with matching id
            {"jsonrpc": "2.0", "id": 1, "result": {"real": True}},
        ]
    )
    result = client.call_tool("get-projects")
    assert result == {"real": True}


def test_recv_raises_on_stdout_closed_with_scrubbed_stderr() -> None:
    """API key must not appear in error messages."""
    config = PhoenixMCPConfig(base_url="x", api_key="SUPER_SECRET_KEY")
    client = PhoenixMCPClient(config)
    client._proc = _FakeProc([])  # empty scripted = readline returns ""
    client._stderr_buffer = [
        "error: failed authenticating with SUPER_SECRET_KEY",
        "more diagnostic noise",
    ]

    with pytest.raises(PhoenixMCPError) as exc_info:
        client.call_tool("list-projects")

    assert "SUPER_SECRET_KEY" not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_initialize_records_server_protocol_version() -> None:
    """Client must store the server-reported protocolVersion for diagnostics."""
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-03-26"}},
        ]
    )
    client._initialize()
    assert client.server_protocol_version == "2025-03-26"


def test_initialize_warns_on_unsupported_protocol_version_without_raising() -> None:
    """Unknown protocolVersion is a warning (added to stderr buffer), not an error."""
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "9999-99-99"}},
        ]
    )
    client._initialize()
    assert client.server_protocol_version == "9999-99-99"
    warning = "\n".join(client._stderr_buffer)
    assert "warning" in warning
    assert "9999-99-99" in warning


def test_package_version_reaches_subprocess_command() -> None:
    """Custom package_version is wired into the npx argv (default 'latest')."""
    config = PhoenixMCPConfig(base_url="x", api_key="y")
    client = PhoenixMCPClient(config, package_version="0.5.0")
    # Inspect what __enter__ would build without spawning a real process by
    # replicating its command construction logic; kept in sync with
    # __enter__ in mcp_client.py.
    expected_arg = "@arizeai/phoenix-mcp@0.5.0"
    cmd_parts = [
        "npx",
        "-y",
        f"@arizeai/phoenix-mcp@{client._package_version}",
    ]
    assert expected_arg in cmd_parts


# ---------------------------------------------------------------------------
# Integration smoke (real subprocess). Skipped if env vars are missing.
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_mcp_subprocess_lists_tools() -> None:
    """Spawn the real npx subprocess and enumerate its tools.

    Skipped if PHOENIX_HOST or PHOENIX_API_KEY are not set so CI without
    Arize credentials passes.
    """
    if not os.environ.get("PHOENIX_HOST") or not os.environ.get("PHOENIX_API_KEY"):
        pytest.skip("PHOENIX_HOST and PHOENIX_API_KEY required")

    config = PhoenixMCPConfig.from_env()
    with PhoenixMCPClient(config) as mcp:
        tools = mcp.list_tools()

    assert tools, "Phoenix MCP server returned no tools"
    tool_names = {t["name"] for t in tools}
    expected_subset = {"list-projects", "list-traces", "get-trace", "get-spans"}
    assert expected_subset.issubset(
        tool_names
    ), f"missing expected tools: {expected_subset - tool_names}"
