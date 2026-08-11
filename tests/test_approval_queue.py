"""Tests for Phase 3.2: the approval queue and resuming a paused task from a human decision."""

import pytest

from orchestra.hitl.approval_queue import ApprovalQueue
from orchestra.hitl.resume import resume_task
from orchestra.orchestration.schemas import ExecutionPlan, ReviewVerdict, SpecialistResult

from tests.helpers import make_step


def _escalated_state(**overrides) -> dict:
    plan = ExecutionPlan(
        reasoning="test",
        confidence=1.0,
        steps=[
            make_step("s1", domain="research", description="find the facts"),
            make_step("s2", domain="writing", depends_on=["s1"], description="write it up"),
        ],
    )
    state = {
        "task": "do the thing",
        "task_id": "t1",
        "user_id": "alice",
        "status": "needs_escalation",
        "plan": plan,
        "step_progress": {"s1": {"attempts": 2, "escalated": True}},
        "completed_step_ids": [],
        "pending_results": [
            SpecialistResult(step_id="s1", domain="research", attempt=2, output="draft facts", confidence=0.4)
        ],
        "specialist_results": [],
        "review_history": [
            ReviewVerdict(step_id="s1", attempt=2, approved=False, confidence=0.4, feedback="too vague")
        ],
        "escalations": [
            {
                "step_id": "s1",
                "attempt": 2,
                "reason": "specialist_failed_twice",
                "feedback": "too vague",
                "confidence": 0.4,
            }
        ],
    }
    state.update(overrides)
    return state


def _plan_level_escalated_state(**overrides) -> dict:
    plan = ExecutionPlan(
        reasoning="not sure", confidence=0.2, steps=[make_step("s1", domain="research")]
    )
    state = {
        "task": "do something uncertain",
        "task_id": "t2",
        "user_id": "alice",
        "status": "needs_escalation",
        "plan": plan,
        "step_progress": {},
        "completed_step_ids": [],
        "pending_results": [],
        "specialist_results": [],
        "review_history": [],
        "escalations": [
            {
                "step_id": "plan",
                "attempt": 0,
                "reason": "low_plan_confidence",
                "feedback": "not sure",
                "confidence": 0.2,
            }
        ],
    }
    state.update(overrides)
    return state


# ---------- ApprovalQueue.push_all ----------


def test_push_all_creates_one_request_per_new_escalation():
    queue = ApprovalQueue()
    requests = queue.push_all(_escalated_state())

    assert len(requests) == 1
    assert requests[0].reason == "specialist_failed_twice"
    assert requests[0].task_id == "t1"
    assert requests[0].user_id == "alice"
    assert requests[0].current_step_id == "s1"
    assert requests[0].proposed_action == "draft facts"
    assert requests[0].status == "pending"


def test_push_all_dedupes_already_queued_escalations():
    queue = ApprovalQueue()
    state = _escalated_state()

    first = queue.push_all(state)
    second = queue.push_all(state)  # same escalations list, e.g. called twice on resume

    assert len(first) == 1
    assert second == []
    assert len(queue.list_pending()) == 1


def test_push_all_plan_level_escalation_has_no_current_step():
    queue = ApprovalQueue()
    requests = queue.push_all(_plan_level_escalated_state())

    assert len(requests) == 1
    assert requests[0].current_step_id is None
    assert requests[0].proposed_action is None
    assert requests[0].reason == "low_plan_confidence"


def test_list_pending_filters_by_user_id():
    queue = ApprovalQueue()
    queue.push_all(_escalated_state(user_id="alice", task_id="t1"))
    queue.push_all(_plan_level_escalated_state(user_id="bob", task_id="t2"))

    assert [r.user_id for r in queue.list_pending(user_id="alice")] == ["alice"]
    assert len(queue.list_pending()) == 2


def test_decide_requires_modified_output_for_modified_decision():
    queue = ApprovalQueue()
    request = queue.push_all(_escalated_state())[0]

    with pytest.raises(ValueError, match="modified_output"):
        queue.decide(request.id, "modified", "someone")


def test_decide_updates_status_and_fields():
    queue = ApprovalQueue()
    request = queue.push_all(_escalated_state())[0]

    updated = queue.decide(request.id, "approved", "alice_pi", reviewer_notes="looks fine")

    assert updated.status == "approved"
    assert updated.decision == "approved"
    assert updated.decided_by == "alice_pi"
    assert updated.reviewer_notes == "looks fine"
    assert updated.decided_at is not None


# ---------- resume_task: step-level ----------


def test_resume_step_level_approved_marks_step_complete_and_continues(monkeypatch):
    from orchestra.agents.base import BaseAgent
    from orchestra.agents.memory_writer import MemoryWriterAgent
    from orchestra.agents.specialists.research import ResearchAgent
    from orchestra.agents.specialists.writing import WritingAgent

    monkeypatch.setattr(ResearchAgent, "tools", [])
    monkeypatch.setattr(WritingAgent, "tools", [])
    monkeypatch.setattr(
        BaseAgent,
        "_call_llm",
        lambda self, system, user, max_tokens=1536: (
            "Final answer." if self.name == "supervisor" else
            '{"approved": true, "confidence": 0.9, "feedback": null}' if self.name == "reviewer" else
            "written output"
        ),
    )
    monkeypatch.setattr(
        MemoryWriterAgent,
        "_call_structured",
        lambda *a, **k: {"approach_summary": "n/a", "domain_facts": [], "user_preferences": []},
    )

    queue = ApprovalQueue()
    request = queue.push_all(_escalated_state())[0]
    queue.decide(request.id, "approved", "alice_pi")

    result = resume_task(request.id, queue=queue)

    assert "s1" in result["completed_step_ids"]
    # s2 depends on s1 and should now run and complete too, finishing the whole plan.
    assert result["status"] == "complete"
    assert result["final_output"] == "Final answer."


def test_resume_step_level_modified_uses_the_reviewers_output(monkeypatch):
    from orchestra.agents.base import BaseAgent
    from orchestra.agents.memory_writer import MemoryWriterAgent
    from orchestra.agents.specialists.research import ResearchAgent
    from orchestra.agents.specialists.writing import WritingAgent

    monkeypatch.setattr(ResearchAgent, "tools", [])
    monkeypatch.setattr(WritingAgent, "tools", [])
    monkeypatch.setattr(
        BaseAgent,
        "_call_llm",
        lambda self, system, user, max_tokens=1536: (
            user if self.name == "writing" else  # so we can assert modified text reached writing
            "Final answer." if self.name == "supervisor" else
            '{"approved": true, "confidence": 0.9, "feedback": null}'
        ),
    )
    monkeypatch.setattr(
        MemoryWriterAgent,
        "_call_structured",
        lambda *a, **k: {"approach_summary": "n/a", "domain_facts": [], "user_preferences": []},
    )

    queue = ApprovalQueue()
    request = queue.push_all(_escalated_state())[0]
    queue.decide(request.id, "modified", "alice_pi", modified_output="corrected facts")

    result = resume_task(request.id, queue=queue)

    s1_result = next(r for r in result["specialist_results"] if r.step_id == "s1")
    assert s1_result.output == "corrected facts"
    assert s1_result.confidence == 1.0


def test_resume_step_level_rejected_retries_with_human_feedback(monkeypatch):
    from orchestra.agents.base import BaseAgent
    from orchestra.agents.memory_writer import MemoryWriterAgent
    from orchestra.agents.specialists.research import ResearchAgent
    from orchestra.agents.specialists.writing import WritingAgent

    captured_prompts = []

    def fake_call_llm(self, system, user, max_tokens=1536):
        if self.name == "research":
            captured_prompts.append(user)
            return "better facts this time"
        if self.name == "writing":
            return "written output"
        if self.name == "supervisor":
            return "Final answer."
        return '{"approved": true, "confidence": 0.9, "feedback": null}'

    monkeypatch.setattr(ResearchAgent, "tools", [])
    monkeypatch.setattr(WritingAgent, "tools", [])
    monkeypatch.setattr(BaseAgent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        MemoryWriterAgent,
        "_call_structured",
        lambda *a, **k: {"approach_summary": "n/a", "domain_facts": [], "user_preferences": []},
    )

    queue = ApprovalQueue()
    request = queue.push_all(_escalated_state())[0]
    queue.decide(request.id, "rejected", "alice_pi", reviewer_notes="cite your sources")

    result = resume_task(request.id, queue=queue)

    assert any("cite your sources" in p for p in captured_prompts)
    assert result["status"] == "complete"


# ---------- resume_task: plan-level ----------


def test_resume_plan_level_approved_runs_the_plan(monkeypatch):
    from orchestra.agents.base import BaseAgent
    from orchestra.agents.memory_writer import MemoryWriterAgent
    from orchestra.agents.specialists.research import ResearchAgent

    monkeypatch.setattr(ResearchAgent, "tools", [])
    monkeypatch.setattr(
        BaseAgent,
        "_call_llm",
        lambda self, system, user, max_tokens=1536: (
            "Final answer." if self.name == "supervisor" else
            '{"approved": true, "confidence": 0.9, "feedback": null}' if self.name == "reviewer" else
            "researched output"
        ),
    )
    monkeypatch.setattr(
        MemoryWriterAgent,
        "_call_structured",
        lambda *a, **k: {"approach_summary": "n/a", "domain_facts": [], "user_preferences": []},
    )

    queue = ApprovalQueue()
    request = queue.push_all(_plan_level_escalated_state())[0]
    queue.decide(request.id, "approved", "alice_pi")

    result = resume_task(request.id, queue=queue)

    assert result["status"] == "complete"


def test_resume_plan_level_rejected_terminates_the_task():
    queue = ApprovalQueue()
    request = queue.push_all(_plan_level_escalated_state())[0]
    queue.decide(request.id, "rejected", "alice_pi", reviewer_notes="not worth doing")

    result = resume_task(request.id, queue=queue)

    assert result["status"] == "rejected"
    assert "final_output" not in result or result["final_output"] is None


def test_resume_plan_level_modified_is_not_supported():
    queue = ApprovalQueue()
    request = queue.push_all(_plan_level_escalated_state())[0]
    queue.decide(request.id, "modified", "alice_pi", modified_output="a different plan")

    with pytest.raises(ValueError, match="only support approve or reject"):
        resume_task(request.id, queue=queue)


def test_resume_raises_for_unknown_request():
    with pytest.raises(ValueError, match="No approval request"):
        resume_task("does-not-exist")


def test_resume_raises_if_not_yet_decided():
    queue = ApprovalQueue()
    request = queue.push_all(_escalated_state())[0]

    with pytest.raises(ValueError, match="hasn't been decided"):
        resume_task(request.id, queue=queue)
