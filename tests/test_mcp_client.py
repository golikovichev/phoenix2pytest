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
import subprocess
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
# Typed wrappers
# ---------------------------------------------------------------------------


def test_list_projects_delegates_to_call_tool() -> None:
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"projects": [{"id": "p1"}]}},
        ]
    )
    result = client.list_projects()

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    assert sent["method"] == "tools/call"
    assert sent["params"]["name"] == "list-projects"
    assert sent["params"]["arguments"] == {}
    assert result == {"projects": [{"id": "p1"}]}


def test_get_trace_passes_trace_id() -> None:
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"trace": {"id": "t1"}}},
        ]
    )
    result = client.get_trace("t1")

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    assert sent["params"]["name"] == "get-trace"
    assert sent["params"]["arguments"] == {"traceId": "t1"}
    assert result == {"trace": {"id": "t1"}}


def test_get_spans_passes_trace_id() -> None:
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"spans": []}},
        ]
    )
    result = client.get_spans("t99")

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    assert sent["params"]["name"] == "get-spans"
    assert sent["params"]["arguments"] == {"traceId": "t99"}
    assert result == {"spans": []}


def test_get_span_annotations_passes_span_id() -> None:
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"annotations": []}},
        ]
    )
    result = client.get_span_annotations("s42")

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    assert sent["params"]["name"] == "get-span-annotations"
    assert sent["params"]["arguments"] == {"spanId": "s42"}
    assert result == {"annotations": []}


def test_list_traces_without_project_or_config_omits_project_id() -> None:
    """No project anywhere = no projectId in arguments."""
    config = PhoenixMCPConfig(base_url="x", api_key="y")  # project=None
    client = PhoenixMCPClient(config)
    client._proc = _FakeProc([{"jsonrpc": "2.0", "id": 1, "result": {"traces": []}}])
    client.list_traces(limit=7)

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    args = sent["params"]["arguments"]
    assert args == {"limit": 7}
    assert "projectId" not in args


# ---------------------------------------------------------------------------
# Discovery + protocol edge cases
# ---------------------------------------------------------------------------


def test_list_tools_returns_tools_array_from_response() -> None:
    client = _client_with_fake(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"tools": [{"name": "list-projects"}, {"name": "get-trace"}]},
            },
        ]
    )
    tools = client.list_tools()

    assert tools == [{"name": "list-projects"}, {"name": "get-trace"}]


def test_list_tools_returns_empty_when_field_missing() -> None:
    """Result without `tools` key still resolves to empty list, not KeyError."""
    client = _client_with_fake(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
        ]
    )
    assert client.list_tools() == []


def test_recv_response_drops_server_initiated_request_with_id_and_method() -> None:
    """Server-initiated request (id + method) is dropped, then real response read."""
    client = _client_with_fake(
        [
            # Server-initiated request shape: id + method together
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "sampling/createMessage",
                "params": {},
            },
            # Real response with matching client request id
            {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}},
        ]
    )
    result = client.call_tool("list-projects")
    assert result == {"ok": True}


def test_scrub_secrets_with_empty_api_key_returns_text_unchanged() -> None:
    """Empty api_key means there is nothing to redact; text comes through as-is."""
    config = PhoenixMCPConfig(base_url="x", api_key="")
    client = PhoenixMCPClient(config)
    out = client._scrub_secrets("nothing to redact here")
    assert out == "nothing to redact here"


def test_scrub_secrets_with_falsy_text_returns_input_unchanged() -> None:
    """Empty input string returns the same empty string."""
    config = PhoenixMCPConfig(base_url="x", api_key="some-key")
    client = PhoenixMCPClient(config)
    assert client._scrub_secrets("") == ""


def test_next_id_increments_monotonically() -> None:
    client = _client_with_fake([])
    first = client._next_id()
    second = client._next_id()
    third = client._next_id()
    assert (first, second, third) == (1, 2, 3)


def test_notify_writes_payload_without_id() -> None:
    """notifications/initialized and friends carry method but no id."""
    client = _client_with_fake([])
    client._notify("notifications/initialized")

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    assert sent == {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert "id" not in sent


def test_notify_writes_params_when_provided() -> None:
    client = _client_with_fake([])
    client._notify("notifications/progress", {"step": 1})

    sent = json.loads(client._proc.stdin._inbox[0])  # type: ignore[union-attr]
    assert sent["method"] == "notifications/progress"
    assert sent["params"] == {"step": 1}
    assert "id" not in sent


# ---------------------------------------------------------------------------
# Lifecycle: __enter__ / __exit__ / _teardown via mocked subprocess.Popen
# ---------------------------------------------------------------------------


class _RecordingFakeProc(_FakeProc):
    """_FakeProc that records terminate/kill/wait calls and stdin close state."""

    def __init__(self, scripted_responses: list[dict]):
        super().__init__(scripted_responses)
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | None] = []

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        return 0


def test_teardown_closes_stdin_terminates_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PhoenixMCPConfig(base_url="x", api_key="y")
    client = PhoenixMCPClient(config)
    proc = _RecordingFakeProc([])
    client._proc = proc  # type: ignore[assignment]

    client._teardown()

    assert proc.stdin.closed is True
    assert proc.terminated is True
    assert proc.wait_calls == [5]
    assert client._proc is None


def test_teardown_is_idempotent_when_no_process() -> None:
    """Calling _teardown without a spawned proc must be a no-op."""
    config = PhoenixMCPConfig(base_url="x", api_key="y")
    client = PhoenixMCPClient(config)
    assert client._proc is None
    client._teardown()  # must not raise
    assert client._proc is None


def test_teardown_kills_on_terminate_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """If terminate->wait times out, _teardown escalates to kill()."""

    class _StubbornProc(_RecordingFakeProc):
        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls.append(timeout)
            # First wait (5s after terminate) times out; second (2s after kill) succeeds
            if len(self.wait_calls) == 1:
                raise subprocess.TimeoutExpired(cmd="x", timeout=timeout or 0)
            return 0

    config = PhoenixMCPConfig(base_url="x", api_key="y")
    client = PhoenixMCPClient(config)
    proc = _StubbornProc([])
    client._proc = proc  # type: ignore[assignment]

    client._teardown()

    assert proc.terminated is True
    assert proc.killed is True
    assert proc.wait_calls == [5, 2]


def test_context_manager_spawns_subprocess_and_handshakes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """__enter__ spawns Popen, sends initialize handshake, returns self."""
    handshake_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"protocolVersion": "2024-11-05"},
    }
    captured: dict = {}

    def fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env", {})
        return _RecordingFakeProc([handshake_response])

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    config = PhoenixMCPConfig(
        base_url="https://phx.test",
        api_key="key-1",
        project="proj-A",
    )
    with PhoenixMCPClient(config, package_version="0.5.0") as mcp:
        assert mcp.server_protocol_version == "2024-11-05"

    assert captured["cmd"][:3] == ["npx", "-y", "@arizeai/phoenix-mcp@0.5.0"]
    assert captured["env"]["PHOENIX_HOST"] == "https://phx.test"
    assert captured["env"]["PHOENIX_API_KEY"] == "key-1"
    assert captured["env"]["PHOENIX_PROJECT"] == "proj-A"


def test_context_manager_omits_project_env_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No project on config means no PHOENIX_PROJECT in child env."""
    captured: dict = {}

    def fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["env"] = kwargs.get("env", {})
        return _RecordingFakeProc(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"protocolVersion": "2024-11-05"},
                }
            ]
        )

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    # Clear any inherited PHOENIX_PROJECT to make the assertion deterministic
    monkeypatch.delenv("PHOENIX_PROJECT", raising=False)

    config = PhoenixMCPConfig(base_url="x", api_key="y")  # project=None
    with PhoenixMCPClient(config) as mcp:
        assert mcp is not None

    assert "PHOENIX_PROJECT" not in captured["env"]


def test_context_manager_teardown_on_handshake_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If _initialize raises, __enter__ still tears the subprocess down."""
    proc = _RecordingFakeProc(
        [
            # Error response on initialize -> raises PhoenixMCPError
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -1, "message": "auth failed"},
            },
        ]
    )

    def fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    config = PhoenixMCPConfig(base_url="x", api_key="y")
    with (
        pytest.raises(PhoenixMCPError, match="auth failed"),
        PhoenixMCPClient(config),
    ):
        pass

    # __enter__ caught the handshake error and ran _teardown
    assert proc.terminated is True


def test_stderr_reader_thread_collects_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """_start_stderr_reader spawns a daemon thread that appends each line."""
    stderr_lines = ["line one\n", "line two\n", "line three\n"]

    proc = _RecordingFakeProc(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"protocolVersion": "2024-11-05"},
            }
        ]
    )
    # Replace stderr with iterable of lines
    proc.stderr = iter(stderr_lines)  # type: ignore[assignment]

    def fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    config = PhoenixMCPConfig(base_url="x", api_key="y")
    with PhoenixMCPClient(config) as mcp:
        # The reader must be a daemon so it cannot block interpreter shutdown
        assert mcp._stderr_thread is not None
        assert mcp._stderr_thread.daemon is True
        # Wait for the iterator to drain
        mcp._stderr_thread.join(timeout=2)
        assert mcp._stderr_buffer == ["line one", "line two", "line three"]


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
    assert expected_subset.issubset(tool_names), (
        f"missing expected tools: {expected_subset - tool_names}"
    )
