"""Tests for the individual Phase 1.3 tool implementations."""

import pytest

from orchestra.tools.implementations.api_call import api_call
from orchestra.tools.implementations.code_execution import code_execution
from orchestra.tools.implementations.database import database_query
from orchestra.tools.implementations.file_io import file_read_write
from orchestra.tools.implementations.web_search import set_search_provider, web_search


# ---------- file_read_write ----------


def test_file_read_write_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRA_WORKSPACE_DIR", str(tmp_path))

    write_result = file_read_write("write", "notes/todo.txt", content="buy milk")
    assert write_result["bytes_written"] == len(b"buy milk")

    read_result = file_read_write("read", "notes/todo.txt")
    assert read_result["content"] == "buy milk"


def test_file_read_write_blocks_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRA_WORKSPACE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="escapes the sandboxed workspace"):
        file_read_write("read", "../../etc/passwd")


def test_file_read_write_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRA_WORKSPACE_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        file_read_write("read", "nope.txt")


def test_file_read_write_unknown_operation_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRA_WORKSPACE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="Unknown operation"):
        file_read_write("delete", "notes/todo.txt")


# ---------- code_execution ----------


def test_code_execution_captures_stdout():
    result = code_execution("print('hello from sandbox')")
    assert "hello from sandbox" in result["stdout"]
    assert result["exit_code"] == 0


def test_code_execution_captures_nonzero_exit():
    result = code_execution("import sys; sys.exit(3)")
    assert result["exit_code"] == 3


def test_code_execution_times_out():
    result = code_execution("import time; time.sleep(5)", timeout_seconds=1)
    assert result["exit_code"] == -1
    assert "timed out" in result["stderr"]


# ---------- database_query ----------


def test_database_query_rejects_non_select():
    with pytest.raises(ValueError, match="read-only SELECT"):
        database_query("DROP TABLE users;")


def test_database_query_rejects_insert():
    with pytest.raises(ValueError, match="read-only SELECT"):
        database_query("INSERT INTO users (name) VALUES ('x');")


# ---------- api_call ----------


def test_api_call_blocks_by_default(monkeypatch):
    monkeypatch.delenv("ORCHESTRA_API_ALLOWLIST", raising=False)
    with pytest.raises(PermissionError, match="not in ORCHESTRA_API_ALLOWLIST"):
        api_call("GET", "https://example.com/data")


def test_api_call_blocks_hosts_outside_allowlist(monkeypatch):
    monkeypatch.setenv("ORCHESTRA_API_ALLOWLIST", "api.trusted.com")
    with pytest.raises(PermissionError):
        api_call("GET", "https://not-trusted.com/data")


# ---------- web_search ----------


def test_web_search_without_provider_raises():
    set_search_provider(None)
    with pytest.raises(NotImplementedError):
        web_search("orchestra multi-agent")


def test_web_search_uses_configured_provider():
    set_search_provider(lambda query, max_results: [{"title": query, "url": "u", "snippet": "s"}])
    try:
        result = web_search("langgraph", max_results=3)
        assert result["results"][0]["title"] == "langgraph"
    finally:
        set_search_provider(None)
