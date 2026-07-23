"""Tests that a specialist's tool-use loop actually calls the registry."""

from dataclasses import dataclass
from types import SimpleNamespace

from orchestra.orchestration.schemas import ExecutionPlan
from orchestra.tools.registry import RateLimit, ToolRegistry, ToolSpec

from tests.helpers import make_step


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list
    stop_reason: str


class FakeAnthropicMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def test_specialist_tool_loop_invokes_registry_and_returns_final_text(monkeypatch):
    from orchestra.agents.specialists.research import ResearchAgent

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="web_search",
            description="search",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            output_schema={"type": "object"},
            allowed_domains=["research"],
            rate_limit=RateLimit(max_calls=10, per_seconds=60),
            handler=lambda query: {"results": [{"title": query}]},
        )
    )
    monkeypatch.setattr("orchestra.agents.specialists.base_specialist.default_registry", registry)
    monkeypatch.setattr(ResearchAgent, "tools", ["web_search"])

    tool_call_response = FakeResponse(
        content=[FakeToolUseBlock(id="t1", name="web_search", input={"query": "orchestra"})],
        stop_reason="tool_use",
    )
    final_response = FakeResponse(
        content=[FakeTextBlock(text="Orchestra is a multi-agent platform.")],
        stop_reason="end_turn",
    )
    fake_messages = FakeAnthropicMessages([tool_call_response, final_response])
    monkeypatch.setattr(
        "orchestra.agents.base.get_client",
        lambda: SimpleNamespace(messages=fake_messages),
    )

    agent = ResearchAgent()
    plan = ExecutionPlan(
        reasoning="test",
        steps=[make_step("s1", domain="research", description="research orchestra")],
    )
    state = {
        "plan": plan,
        "current_step_index": 0,
        "specialist_results": [],
        "review_history": [],
    }

    update = agent(state)

    assert update["pending_result"].output == "Orchestra is a multi-agent platform."
    assert update["pending_result"].tool_calls == ["web_search"]
    assert len(registry.invocation_log) == 1
    assert registry.invocation_log[0].tool_name == "web_search"
    assert registry.invocation_log[0].success is True
