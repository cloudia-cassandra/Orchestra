"""Tests for Phase 1.2: the task decomposition engine."""

import pytest
from pydantic import ValidationError

from orchestra.orchestration.schemas import ExecutionPlan

from tests.helpers import make_step


def test_valid_dag_passes():
    plan = ExecutionPlan(
        reasoning="two independent lookups feeding a writeup",
        steps=[
            make_step("s1", domain="research"),
            make_step("s2", domain="data_analysis"),
            make_step("s3", domain="writing", depends_on=["s1", "s2"]),
        ],
    )
    assert [s.step_id for s in plan.steps] == ["s1", "s2", "s3"]


def test_rejects_duplicate_step_ids():
    with pytest.raises(ValidationError, match="Duplicate step_id"):
        ExecutionPlan(
            reasoning="bad plan",
            steps=[make_step("s1"), make_step("s1")],
        )


def test_rejects_dependency_on_unknown_step():
    with pytest.raises(ValidationError, match="unknown step"):
        ExecutionPlan(
            reasoning="bad plan",
            steps=[make_step("s1", depends_on=["ghost"])],
        )


def test_rejects_self_dependency():
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        ExecutionPlan(
            reasoning="bad plan",
            steps=[make_step("s1", depends_on=["s1"])],
        )


def test_rejects_forward_reference_dependency():
    # s1 depends on s2, but s2 comes later in the list — dependencies must be satisfied
    # by the time the plan reaches the dependent step.
    with pytest.raises(ValidationError, match="must appear earlier"):
        ExecutionPlan(
            reasoning="bad plan",
            steps=[
                make_step("s1", depends_on=["s2"]),
                make_step("s2"),
            ],
        )


def test_rejects_empty_plan():
    with pytest.raises(ValidationError, match="at least one step"):
        ExecutionPlan(reasoning="empty", steps=[])


def test_supervisor_plan_uses_structured_tool_call(monkeypatch):
    from orchestra.agents.supervisor import SupervisorAgent

    agent = SupervisorAgent()
    captured = {}

    def fake_structured(system, user, tool_name, tool_description, input_schema, max_tokens=2048):
        captured["tool_name"] = tool_name
        return {
            "reasoning": "split into research then write",
            "steps": [
                {
                    "step_id": "s1",
                    "domain": "research",
                    "description": "find the facts",
                    "depends_on": [],
                    "required_inputs": [],
                    "expected_output_format": "bullet list of facts",
                    "estimated_complexity": "low",
                },
                {
                    "step_id": "s2",
                    "domain": "writing",
                    "description": "write it up",
                    "depends_on": ["s1"],
                    "required_inputs": ["s1.output: facts"],
                    "expected_output_format": "one paragraph",
                    "estimated_complexity": "low",
                },
            ],
        }

    monkeypatch.setattr(agent, "_call_structured", fake_structured)

    update = agent({"task": "explain something"})

    assert captured["tool_name"] == "submit_execution_plan"
    assert update["status"] == "delegating"
    assert update["current_step_index"] == 0
    assert [s.step_id for s in update["plan"].steps] == ["s1", "s2"]


def test_supervisor_retries_on_invalid_plan_then_succeeds(monkeypatch):
    from orchestra.agents.supervisor import SupervisorAgent

    agent = SupervisorAgent()
    calls = {"count": 0}

    invalid = {
        "reasoning": "broken",
        "steps": [
            {
                "step_id": "s1",
                "domain": "research",
                "description": "find the facts",
                "depends_on": ["ghost"],
                "required_inputs": [],
                "expected_output_format": "text",
                "estimated_complexity": "low",
            }
        ],
    }
    valid = {
        "reasoning": "fixed",
        "steps": [
            {
                "step_id": "s1",
                "domain": "research",
                "description": "find the facts",
                "depends_on": [],
                "required_inputs": [],
                "expected_output_format": "text",
                "estimated_complexity": "low",
            }
        ],
    }

    def fake_structured(system, user, tool_name, tool_description, input_schema, max_tokens=2048):
        calls["count"] += 1
        return invalid if calls["count"] == 1 else valid

    monkeypatch.setattr(agent, "_call_structured", fake_structured)

    update = agent({"task": "explain something"})

    assert calls["count"] == 2
    assert update["plan"].steps[0].step_id == "s1"


def test_specialist_prompt_includes_dependency_outputs():
    from orchestra.agents.specialists.writing import WritingAgent
    from orchestra.orchestration.schemas import ExecutionPlan, SpecialistResult

    agent = WritingAgent()
    plan = ExecutionPlan(
        reasoning="test",
        steps=[
            make_step("s1", domain="research", expected_output_format="bullets"),
            make_step(
                "s2",
                domain="writing",
                depends_on=["s1"],
                required_inputs=["s1.output: key facts"],
                expected_output_format="one paragraph",
            ),
        ],
    )
    state = {
        "plan": plan,
        "current_step_index": 1,
        "specialist_results": [
            SpecialistResult(step_id="s1", domain="research", output="Fact A. Fact B.", confidence=0.9)
        ],
        "review_history": [],
    }

    captured = {}
    expected_output = "final prose"

    def fake_llm(system, user, max_tokens=1536):
        captured["user"] = user
        return expected_output

    agent._call_llm = fake_llm
    update = agent(state)

    assert "Fact A. Fact B." in captured["user"]
    assert "s1.output: key facts" in captured["user"]
    assert "one paragraph" in captured["user"]
    assert update["pending_result"].output == expected_output
