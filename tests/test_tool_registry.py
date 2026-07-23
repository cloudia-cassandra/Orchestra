"""Tests for the generic ToolRegistry: registration, authorization, rate limiting, logging."""

import pytest

from orchestra.tools.registry import RateLimit, RateLimitExceeded, ToolRegistry, ToolSpec


def _make_spec(name="echo", domains=None, max_calls=5, per_seconds=60, handler=None):
    return ToolSpec(
        name=name,
        description="echoes its input",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"value": {"type": "string"}}},
        allowed_domains=domains or ["research"],
        rate_limit=RateLimit(max_calls=max_calls, per_seconds=per_seconds),
        handler=handler or (lambda value: {"value": value}),
    )


def test_register_and_get():
    registry = ToolRegistry()
    spec = _make_spec()
    registry.register(spec)
    assert registry.get("echo") is spec


def test_register_duplicate_raises():
    registry = ToolRegistry()
    registry.register(_make_spec())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_make_spec())


def test_get_unknown_tool_raises():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        registry.get("nope")


def test_list_for_domain_filters_correctly():
    registry = ToolRegistry()
    registry.register(_make_spec(name="research_tool", domains=["research"]))
    registry.register(_make_spec(name="writing_tool", domains=["writing"]))
    assert [t.name for t in registry.list_for_domain("research")] == ["research_tool"]


def test_invoke_success_is_logged():
    registry = ToolRegistry()
    registry.register(_make_spec())
    output = registry.invoke("echo", domain="research", value="hi")
    assert output == {"value": "hi"}

    record = registry.invocation_log[-1]
    assert record.tool_name == "echo"
    assert record.domain == "research"
    assert record.inputs == {"value": "hi"}
    assert record.output == {"value": "hi"}
    assert record.success is True
    assert record.latency_ms >= 0


def test_invoke_failure_is_logged_and_reraised():
    def boom(value):
        raise RuntimeError("kaboom")

    registry = ToolRegistry()
    registry.register(_make_spec(handler=boom))

    with pytest.raises(RuntimeError, match="kaboom"):
        registry.invoke("echo", domain="research", value="hi")

    record = registry.invocation_log[-1]
    assert record.success is False
    assert record.error == "kaboom"
    assert record.output is None


def test_invoke_rejects_unauthorized_domain():
    registry = ToolRegistry()
    registry.register(_make_spec(domains=["research"]))
    with pytest.raises(PermissionError, match="not permitted"):
        registry.invoke("echo", domain="writing", value="hi")


def test_invoke_enforces_rate_limit():
    registry = ToolRegistry()
    registry.register(_make_spec(max_calls=2, per_seconds=60))

    registry.invoke("echo", domain="research", value="1")
    registry.invoke("echo", domain="research", value="2")
    with pytest.raises(RateLimitExceeded):
        registry.invoke("echo", domain="research", value="3")
