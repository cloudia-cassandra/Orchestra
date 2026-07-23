"""Structural checks for the Phase 1 agent graph — no LLM calls, no API key needed."""

from orchestra.orchestration.graph import SPECIALIST_DOMAINS, build_graph
from orchestra.orchestration.schemas import ExecutionPlan, ReviewVerdict
from orchestra.orchestration.state import OrchestraState

from tests.helpers import make_step


def test_graph_compiles_with_expected_nodes():
    compiled = build_graph()
    node_names = set(compiled.get_graph().nodes)
    expected = {"supervisor", "reviewer", *SPECIALIST_DOMAINS}
    assert expected.issubset(node_names)


def test_route_after_supervisor_picks_current_step_domain():
    from orchestra.orchestration.graph import route_after_supervisor

    plan = ExecutionPlan(reasoning="test", steps=[make_step("s1", domain="writing")])
    state: OrchestraState = {"plan": plan, "current_step_index": 0, "status": "delegating"}
    assert route_after_supervisor(state) == "writing"


def test_route_after_supervisor_ends_when_complete():
    from orchestra.orchestration.graph import route_after_supervisor

    state: OrchestraState = {"status": "complete"}
    assert route_after_supervisor(state) == "end"


def test_route_after_review_retries_same_domain():
    from orchestra.orchestration.graph import route_after_review

    plan = ExecutionPlan(reasoning="test", steps=[make_step("s1", domain="research")])
    state: OrchestraState = {"plan": plan, "current_step_index": 0, "status": "retrying"}
    assert route_after_review(state) == "research"


def test_route_after_review_escalates_after_max_retries():
    from orchestra.orchestration.graph import route_after_review

    state: OrchestraState = {"status": "needs_escalation"}
    assert route_after_review(state) == "end"


def test_route_after_review_returns_to_supervisor_on_approval():
    from orchestra.orchestration.graph import route_after_review

    state: OrchestraState = {"status": "delegating"}
    assert route_after_review(state) == "supervisor"


def test_reviewer_rejects_and_increments_retry_count(monkeypatch):
    from orchestra.agents.reviewer import ReviewerAgent
    from orchestra.orchestration.schemas import SpecialistResult

    agent = ReviewerAgent()
    monkeypatch.setattr(
        agent,
        "_call_llm",
        lambda system, user, max_tokens=1536: '{"approved": false, "confidence": 0.4, "feedback": "too vague"}',
    )

    plan = ExecutionPlan(reasoning="test", steps=[make_step("s1", domain="writing")])
    result = SpecialistResult(step_id="s1", domain="writing", output="...", confidence=0.5)
    state: OrchestraState = {
        "plan": plan,
        "current_step_index": 0,
        "pending_result": result,
        "retry_count": 0,
        "review_history": [],
    }

    update = agent(state)
    assert update["status"] == "retrying"
    assert update["retry_count"] == 1
    assert update["review_history"][0].approved is False


def test_reviewer_escalates_after_max_retries(monkeypatch):
    from orchestra.agents.reviewer import MAX_RETRIES, ReviewerAgent
    from orchestra.orchestration.schemas import SpecialistResult

    agent = ReviewerAgent()
    monkeypatch.setattr(
        agent,
        "_call_llm",
        lambda system, user, max_tokens=1536: '{"approved": false, "confidence": 0.2, "feedback": "still wrong"}',
    )

    plan = ExecutionPlan(reasoning="test", steps=[make_step("s1", domain="writing")])
    result = SpecialistResult(step_id="s1", domain="writing", output="...", confidence=0.5)
    state: OrchestraState = {
        "plan": plan,
        "current_step_index": 0,
        "pending_result": result,
        "retry_count": MAX_RETRIES,
        "review_history": [ReviewVerdict(step_id="s1", approved=False, confidence=0.3)],
    }

    update = agent(state)
    assert update["status"] == "needs_escalation"
