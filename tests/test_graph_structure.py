"""Structural checks for the Phase 1.4 agent graph — no LLM calls, no API key needed."""

from langgraph.types import Send

from orchestra.orchestration.graph import (
    SPECIALIST_DOMAINS,
    build_graph,
    route_after_memory_writer,
    route_after_reviewer,
    route_after_supervisor,
)
from orchestra.orchestration.schemas import ExecutionPlan
from orchestra.orchestration.state import OrchestraState

from tests.helpers import make_step


def test_graph_compiles_with_expected_nodes():
    compiled = build_graph()
    node_names = set(compiled.get_graph().nodes)
    expected = {"intake", "supervisor", "reviewer", "memory_writer", "delivery", *SPECIALIST_DOMAINS}
    assert expected.issubset(node_names)


def test_route_after_supervisor_ends_when_complete():
    state: OrchestraState = {"status": "complete"}
    assert route_after_supervisor(state) == "memory_writer"


def test_route_after_supervisor_routes_escalation_to_memory_writer():
    state: OrchestraState = {"status": "needs_escalation"}
    assert route_after_supervisor(state) == "memory_writer"


def test_route_after_supervisor_fans_out_to_ready_steps():
    plan = ExecutionPlan(
        reasoning="test",
        steps=[
            make_step("s1", domain="research"),
            make_step("s2", domain="writing", depends_on=["s1"]),
        ],
    )
    state: OrchestraState = {
        "status": "executing",
        "plan": plan,
        "completed_step_ids": [],
        "step_progress": {},
    }

    result = route_after_supervisor(state)

    assert isinstance(result, list)
    assert len(result) == 1
    send = result[0]
    assert isinstance(send, Send)
    assert send.node == "research"
    assert send.arg["active_step_id"] == "s1"


def test_route_after_supervisor_fans_out_to_multiple_independent_steps_in_parallel():
    plan = ExecutionPlan(
        reasoning="test",
        steps=[
            make_step("s1", domain="research"),
            make_step("s2", domain="data_analysis"),
        ],
    )
    state: OrchestraState = {
        "status": "executing",
        "plan": plan,
        "completed_step_ids": [],
        "step_progress": {},
    }

    result = route_after_supervisor(state)

    assert {send.node for send in result} == {"research", "data_analysis"}
    assert {send.arg["active_step_id"] for send in result} == {"s1", "s2"}


def test_route_after_supervisor_ends_on_deadlock():
    # A step that's neither completed nor escalated but also not ready (impossible with a
    # valid DAG, but the router should fail loudly rather than loop forever).
    plan = ExecutionPlan(reasoning="test", steps=[make_step("s1", depends_on=[])])
    state: OrchestraState = {
        "status": "executing",
        "plan": plan,
        "completed_step_ids": [],
        "step_progress": {"s1": {"escalated": True}},
    }
    assert route_after_supervisor(state) == "end"


def test_route_after_reviewer_routes_escalation_to_memory_writer():
    assert route_after_reviewer({"status": "needs_escalation"}) == "memory_writer"


def test_route_after_reviewer_returns_to_supervisor_otherwise():
    assert route_after_reviewer({"status": "executing"}) == "supervisor"


def test_route_after_memory_writer_goes_to_delivery_when_complete():
    assert route_after_memory_writer({"status": "complete"}) == "delivery"


def test_route_after_memory_writer_routes_to_approval_queue_without_delivery_when_escalated():
    assert route_after_memory_writer({"status": "needs_escalation"}) == "approval_queue"
