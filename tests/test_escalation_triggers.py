"""Tests for Phase 3.1: escalation triggers — the supervisor's plan-level triggers (user
requested, sensitive operation, low plan confidence) and the sensitivity classifier they use.
The reviewer's two triggers (specialist failed twice, low quality score) are covered in
test_reviewer.py; this file covers everything new in Phase 3.1."""

from orchestra.agents.base import BaseAgent
from orchestra.agents.memory_writer import MemoryWriterAgent
from orchestra.agents.specialists.research import ResearchAgent
from orchestra.agents.specialists.base_specialist import SpecialistAgent
from orchestra.agents.supervisor import SupervisorAgent
from orchestra.hitl.triggers import classify_sensitivity
from orchestra.orchestration.graph import build_graph
from orchestra.orchestration.schemas import ExecutionPlan

from tests.helpers import make_step

_MEMORY_EXTRACTION = {
    "approach_summary": "n/a",
    "domain_facts": [],
    "user_preferences": [],
}


# ---------- classify_sensitivity ----------


def test_classify_sensitivity_flags_financial_language():
    step = make_step("s1", description="Process the customer's payment for the invoice.")
    assert classify_sensitivity(step) == "financial_transaction"


def test_classify_sensitivity_flags_deletion_language():
    step = make_step("s1", description="Delete the outdated records from the archive.")
    assert classify_sensitivity(step) == "data_deletion"


def test_classify_sensitivity_flags_communication_language():
    step = make_step("s1", description="Send an email to the client confirming the change.")
    assert classify_sensitivity(step) == "external_communication"


def test_classify_sensitivity_returns_none_for_ordinary_step():
    step = make_step("s1", description="Summarize the attached PDF into bullet points.")
    assert classify_sensitivity(step) is None


def test_classify_sensitivity_checks_required_inputs_too():
    step = make_step("s1", description="Follow up as discussed.", required_inputs=["wire transfer confirmation number"])
    assert classify_sensitivity(step) == "financial_transaction"


# ---------- SupervisorAgent._check_plan_triggers ----------


def _plan(confidence=1.0, steps=None):
    return ExecutionPlan(
        reasoning="test plan",
        confidence=confidence,
        steps=steps or [make_step("s1", description="research the topic")],
    )


def test_check_plan_triggers_returns_none_for_an_ordinary_confident_plan():
    agent = SupervisorAgent()
    assert agent._check_plan_triggers({}, _plan(confidence=0.95)) is None


def test_check_plan_triggers_flags_low_confidence():
    agent = SupervisorAgent()
    escalation = agent._check_plan_triggers({}, _plan(confidence=0.2))
    assert escalation["reason"] == "low_plan_confidence"
    assert escalation["step_id"] == "plan"


def test_check_plan_triggers_flags_sensitive_step_even_with_high_confidence():
    agent = SupervisorAgent()
    sensitive_step = make_step("s1", description="Wire transfer funds to the vendor account.")
    escalation = agent._check_plan_triggers({}, _plan(confidence=0.99, steps=[sensitive_step]))
    assert escalation["reason"] == "sensitive_operation"
    assert escalation["step_id"] == "s1"


def test_check_plan_triggers_user_requested_wins_over_everything_else():
    agent = SupervisorAgent()
    escalation = agent._check_plan_triggers(
        {"user_requested_review": True}, _plan(confidence=0.99)
    )
    assert escalation["reason"] == "user_requested"


# ---------- SupervisorAgent._plan end-to-end (mocked LLM) ----------


def test_plan_escalates_before_execution_when_confidence_is_low(monkeypatch):
    agent = SupervisorAgent()
    monkeypatch.setattr(
        agent,
        "_call_structured",
        lambda **kwargs: {
            "reasoning": "not sure this will work",
            "confidence": 0.1,
            "steps": [
                {
                    "step_id": "s1",
                    "domain": "research",
                    "description": "find the facts",
                    "depends_on": [],
                    "required_inputs": [],
                    "expected_output_format": "bullets",
                    "estimated_complexity": "low",
                }
            ],
        },
    )

    update = agent({"task": "do something uncertain", "task_id": "t1"})

    assert update["status"] == "needs_escalation"
    assert update["escalations"][0]["reason"] == "low_plan_confidence"
    assert update["plan"] is not None  # still recorded, just not executed


# ---------- full graph: escalation short-circuits before any specialist runs ----------


def test_user_requested_review_short_circuits_before_any_specialist_dispatch(monkeypatch):
    dispatch_order: list[str] = []
    original_call = SpecialistAgent.__call__

    def recording_call(self, state):
        dispatch_order.append(state["active_step_id"])
        return original_call(self, state)

    monkeypatch.setattr(SpecialistAgent, "__call__", recording_call)
    monkeypatch.setattr(ResearchAgent, "tools", [])
    monkeypatch.setattr(BaseAgent, "_call_llm", lambda self, system, user, max_tokens=1536: "unused")
    monkeypatch.setattr(
        SupervisorAgent,
        "_call_structured",
        lambda *a, **k: {
            "reasoning": "single step",
            "confidence": 0.95,
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
        },
    )
    monkeypatch.setattr(MemoryWriterAgent, "_call_structured", lambda *a, **k: _MEMORY_EXTRACTION)

    graph = build_graph()
    result = graph.invoke(
        {"task": "do something", "user_requested_review": True}, config={"recursion_limit": 50}
    )

    assert result["status"] == "needs_escalation"
    assert result["escalations"][0]["reason"] == "user_requested"
    assert dispatch_order == []  # no specialist ever ran

    from orchestra.memory.redis_client import get_redis_client
    from orchestra.memory.working_memory import WorkingMemory

    # Working memory survives — the escalation path skips delivery, same as Phase 2.3.
    assert get_redis_client().get(WorkingMemory(result["task_id"])._key("task")) is not None
