"""Tests for the Phase 1.4 batched Reviewer Agent: approval, rejection, and both escalation
triggers (max attempts exhausted, low confidence)."""

from orchestra.agents.reviewer import MAX_ATTEMPTS, ReviewerAgent
from orchestra.orchestration.schemas import ExecutionPlan, SpecialistResult

from tests.helpers import make_step


def _agent_with_verdict(verdict_json: str) -> ReviewerAgent:
    agent = ReviewerAgent()
    agent._call_llm = lambda system, user, max_tokens=1536: verdict_json
    return agent


def test_approval_marks_step_completed():
    agent = _agent_with_verdict('{"approved": true, "confidence": 0.9, "feedback": null}')
    plan = ExecutionPlan(reasoning="t", steps=[make_step("s1")])
    result = SpecialistResult(step_id="s1", domain="research", attempt=1, output="ok", confidence=0.8)

    update = agent({"plan": plan, "pending_results": [result], "review_history": []})

    assert update["completed_step_ids"] == ["s1"]
    assert update["specialist_results"] == [result]
    assert update["escalations"] == []
    assert update["status"] != "needs_escalation"


def test_rejection_below_max_attempts_does_not_escalate():
    agent = _agent_with_verdict('{"approved": false, "confidence": 0.9, "feedback": "too short"}')
    plan = ExecutionPlan(reasoning="t", steps=[make_step("s1")])
    result = SpecialistResult(step_id="s1", domain="research", attempt=1, output="ok", confidence=0.8)

    update = agent({"plan": plan, "pending_results": [result], "review_history": []})

    assert update["completed_step_ids"] == []
    assert update["escalations"] == []
    assert update["status"] != "needs_escalation"
    assert update["review_history"][0].approved is False


def test_rejection_at_max_attempts_escalates():
    agent = _agent_with_verdict('{"approved": false, "confidence": 0.9, "feedback": "still wrong"}')
    plan = ExecutionPlan(reasoning="t", steps=[make_step("s1")])
    result = SpecialistResult(
        step_id="s1", domain="research", attempt=MAX_ATTEMPTS, output="ok", confidence=0.8
    )

    update = agent({"plan": plan, "pending_results": [result], "review_history": []})

    assert update["status"] == "needs_escalation"
    assert update["step_progress"] == {"s1": {"escalated": True}}
    assert update["escalations"][0]["reason"] == "max_attempts_exceeded"
    assert update["completed_step_ids"] == []


def test_low_confidence_escalates_even_if_approved():
    agent = _agent_with_verdict('{"approved": true, "confidence": 0.2, "feedback": null}')
    plan = ExecutionPlan(reasoning="t", steps=[make_step("s1")])
    result = SpecialistResult(step_id="s1", domain="research", attempt=1, output="ok", confidence=0.8)

    update = agent({"plan": plan, "pending_results": [result], "review_history": []})

    assert update["status"] == "needs_escalation"
    assert update["escalations"][0]["reason"] == "low_confidence"
    assert update["completed_step_ids"] == []


def test_already_reviewed_attempt_is_skipped():
    from orchestra.orchestration.schemas import ReviewVerdict

    agent = ReviewerAgent()
    agent._call_llm = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called"))
    plan = ExecutionPlan(reasoning="t", steps=[make_step("s1")])
    result = SpecialistResult(step_id="s1", domain="research", attempt=1, output="ok", confidence=0.8)
    prior_verdict = ReviewVerdict(step_id="s1", attempt=1, approved=True, confidence=0.9)

    update = agent(
        {"plan": plan, "pending_results": [result], "review_history": [prior_verdict]}
    )

    assert update["review_history"] == []
    assert update["completed_step_ids"] == []


def test_batches_multiple_pending_results_in_one_call():
    agent = ReviewerAgent()
    responses = iter(
        [
            '{"approved": true, "confidence": 0.9, "feedback": null}',
            '{"approved": false, "confidence": 0.9, "feedback": "nope"}',
        ]
    )
    agent._call_llm = lambda *a, **k: next(responses)

    plan = ExecutionPlan(
        reasoning="t", steps=[make_step("s1", domain="research"), make_step("s2", domain="writing")]
    )
    r1 = SpecialistResult(step_id="s1", domain="research", attempt=1, output="a", confidence=0.8)
    r2 = SpecialistResult(step_id="s2", domain="writing", attempt=1, output="b", confidence=0.8)

    update = agent({"plan": plan, "pending_results": [r1, r2], "review_history": []})

    assert update["completed_step_ids"] == ["s1"]
    assert len(update["review_history"]) == 2
