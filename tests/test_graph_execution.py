"""End-to-end graph execution, fully mocked at the LLM boundary — no API key needed.

Proves the wave-based scheduler actually does the thing Phase 1.4 asked for: independent
steps dispatch together (parallel), a step with a dependency waits for it (sequential), and
the whole thing reaches synthesis and delivery.
"""

from orchestra.agents.base import BaseAgent
from orchestra.agents.specialists.code_execution import CodeExecutionAgent
from orchestra.agents.specialists.data_analysis import DataAnalysisAgent
from orchestra.agents.specialists.research import ResearchAgent
from orchestra.agents.specialists.writing import WritingAgent
from orchestra.agents.supervisor import SupervisorAgent
from orchestra.orchestration.graph import build_graph

_PLAN = {
    "reasoning": "two independent lookups feed a writeup",
    "steps": [
        {
            "step_id": "s1",
            "domain": "research",
            "description": "research the topic",
            "depends_on": [],
            "required_inputs": [],
            "expected_output_format": "bullets",
            "estimated_complexity": "low",
        },
        {
            "step_id": "s2",
            "domain": "data_analysis",
            "description": "crunch the numbers",
            "depends_on": [],
            "required_inputs": [],
            "expected_output_format": "table",
            "estimated_complexity": "low",
        },
        {
            "step_id": "s3",
            "domain": "writing",
            "description": "write it up",
            "depends_on": ["s1", "s2"],
            "required_inputs": ["s1.output", "s2.output"],
            "expected_output_format": "paragraph",
            "estimated_complexity": "low",
        },
    ],
}


def _fake_call_llm(self, system, user, max_tokens=1536):
    if self.name == "supervisor":
        return "Final synthesized answer."
    if self.name == "reviewer":
        return '{"approved": true, "confidence": 0.9, "feedback": null}'
    return f"Result from {self.name}"


def test_full_run_dispatches_independent_steps_in_parallel_then_the_dependent_step(monkeypatch):
    dispatch_order: list[str] = []

    monkeypatch.setattr(BaseAgent, "_call_llm", _fake_call_llm)
    monkeypatch.setattr(SupervisorAgent, "_call_structured", lambda *a, **k: _PLAN)

    for agent_cls in (ResearchAgent, DataAnalysisAgent, WritingAgent, CodeExecutionAgent):
        monkeypatch.setattr(agent_cls, "tools", [])

    # Track which active_step_id each specialist node actually receives, in call order, so we
    # can confirm s1/s2 land in the same wave and s3 only shows up after both are done.
    from orchestra.agents.specialists.base_specialist import SpecialistAgent

    original_specialist_call = SpecialistAgent.__call__

    def recording_call(self, state):
        dispatch_order.append(state["active_step_id"])
        return original_specialist_call(self, state)

    monkeypatch.setattr(SpecialistAgent, "__call__", recording_call)

    graph = build_graph()
    result = graph.invoke({"task": "write a report"}, config={"recursion_limit": 50})

    assert result["status"] == "complete"
    assert result["final_output"] == "Final synthesized answer."
    assert set(result["completed_step_ids"]) == {"s1", "s2", "s3"}
    assert len(result["specialist_results"]) == 3

    # s1 and s2 have no dependencies on each other, so both must be dispatched before s3,
    # which depends on both.
    s3_index = dispatch_order.index("s3")
    assert set(dispatch_order[:s3_index]) == {"s1", "s2"}


def test_full_run_escalates_when_reviewer_never_approves(monkeypatch):
    plan_one_step = {
        "reasoning": "single step",
        "steps": [
            {
                "step_id": "s1",
                "domain": "research",
                "description": "research the topic",
                "depends_on": [],
                "required_inputs": [],
                "expected_output_format": "bullets",
                "estimated_complexity": "low",
            }
        ],
    }

    def always_reject(self, system, user, max_tokens=1536):
        if self.name == "reviewer":
            return '{"approved": false, "confidence": 0.9, "feedback": "not good enough"}'
        return "some output"

    monkeypatch.setattr(BaseAgent, "_call_llm", always_reject)
    monkeypatch.setattr(SupervisorAgent, "_call_structured", lambda *a, **k: plan_one_step)
    monkeypatch.setattr(ResearchAgent, "tools", [])

    graph = build_graph()
    result = graph.invoke({"task": "do something hard"}, config={"recursion_limit": 50})

    assert result["status"] == "needs_escalation"
    assert result["escalations"][0]["step_id"] == "s1"
    assert result["escalations"][0]["reason"] == "max_attempts_exceeded"
    assert "final_output" not in result or result["final_output"] is None
