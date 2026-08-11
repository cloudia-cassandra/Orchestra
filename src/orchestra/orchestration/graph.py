"""Wires the intake / supervisor / specialist / reviewer / delivery nodes into a LangGraph
StateGraph.

Pipeline: task intake -> planning -> parallel/sequential specialist execution -> review ->
synthesis -> long-term memory extraction -> delivery (or, on escalation, the approval queue).

- Execution is wave-based: after planning (and after every review), the supervisor's
  conditional edge fans out via LangGraph `Send` to every step whose dependencies are already
  satisfied (orchestration/waves.ready_steps). Independent steps run in the same wave, in
  parallel; dependent steps simply aren't ready until an earlier wave completes them — this is
  what gives us "parallel where possible, sequential where required" from one mechanism.
- If a reviewer rejects a step's output, that step is neither completed nor escalated, so the
  next wave's `ready_steps` call naturally re-dispatches it to the same specialist — which sees
  the rejection feedback in review_history and is prompted to try a different approach.
- If a step's specialist node raises (transient failure — rate limit, network blip), LangGraph's
  retry_policy retries the node itself with backoff before giving up.
- If any Phase 3.1 escalation trigger fires, the run halts in "needs_escalation" and the
  `approval_queue` node (Phase 3.2) packages the full context and pushes it to Postgres for a
  human — see hitl/approval_queue.py and hitl/resume.py, the other half: applying a human's
  decision and calling `build_graph().invoke()` again to continue from exactly this point,
  since `intake_node` and `SupervisorAgent` both already know not to re-plan or reset progress
  when a plan is already present in the state they're handed.
"""

import uuid

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, Send

from orchestra.agents.memory_writer import MemoryWriterAgent
from orchestra.agents.reviewer import ReviewerAgent
from orchestra.agents.specialists.code_execution import CodeExecutionAgent
from orchestra.agents.specialists.data_analysis import DataAnalysisAgent
from orchestra.agents.specialists.research import ResearchAgent
from orchestra.agents.specialists.writing import WritingAgent
from orchestra.agents.supervisor import SupervisorAgent
from orchestra.hitl.approval_queue import ApprovalQueue
from orchestra.hitl.notifications import notify_reviewer
from orchestra.memory.working_memory import WorkingMemory
from orchestra.orchestration.state import OrchestraState
from orchestra.orchestration.waves import ready_steps

SPECIALIST_DOMAINS = ("research", "data_analysis", "writing", "code_execution")

# Retries transient specialist failures (API hiccups, rate limits) before giving up. This is
# distinct from reviewer-driven retries: this handles the node *crashing*, not the node
# succeeding with output the reviewer doesn't like.
SPECIALIST_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=1.0)


DEFAULT_USER_ID = "default_user"


def intake_node(state: OrchestraState) -> dict:
    task = (state.get("task") or "").strip()
    if not task:
        raise ValueError("OrchestraState.task must be a non-empty string.")

    task_id = state.get("task_id") or uuid.uuid4().hex
    # No auth system exists yet, so an unspecified caller is attributed to a shared default
    # user rather than left blank — long-term memory (Phase 2.4) is scoped by user_id, and an
    # empty/missing value would otherwise silently pool every caller's memory together.
    user_id = state.get("user_id") or DEFAULT_USER_ID
    WorkingMemory(task_id).set_task(task)

    update = {"task_id": task_id, "user_id": user_id}
    if state.get("plan") is None:
        update["status"] = "planning"
    # else: this is a resumed task (hitl/resume.py) re-entering at START — a plan already
    # exists, so leave whatever status resume.py set (executing/rejected) alone instead of
    # clobbering it back to "planning".
    return update


def delivery_node(state: OrchestraState) -> dict:
    # The task succeeded end to end — its working memory has done its job, so let it go.
    # A task that instead halts in needs_escalation/rejected skips this node entirely, leaving
    # its memory in Redis for whoever picks up the escalation to inspect.
    WorkingMemory(state["task_id"]).clear()
    return {}


def approval_queue_node(state: OrchestraState) -> dict:
    # Phase 3.2: package the full context and push it to the review queue, then notify. One
    # ApprovalRequest per new escalation (ApprovalQueue.push_all dedupes against ones already
    # queued, since a resumed-and-re-escalated task's snapshot carries its earlier ones too).
    for request in ApprovalQueue().push_all(state):
        notify_reviewer(request)
    return {}


def route_after_supervisor(state: OrchestraState) -> str | list[Send]:
    if state["status"] == "complete":
        return "memory_writer"
    if state["status"] == "needs_escalation":
        # Escalated tasks still pass through memory_writer — Phase 2.3 wants failed approaches
        # remembered too, not just successful ones.
        return "memory_writer"
    if state["status"] == "rejected":
        # A resumed plan-level escalation the human rejected — nothing to run, nothing new to
        # remember (memory_writer already ran once for this task at the original escalation).
        return "end"

    ready = ready_steps(state)
    if not ready:
        # A well-formed DAG with steps still incomplete always has a ready step. Getting here
        # means something upstream is inconsistent — surface it rather than looping forever.
        return "end"
    return [Send(step.domain, {**state, "active_step_id": step.step_id}) for step in ready]


def route_after_reviewer(state: OrchestraState) -> str:
    if state["status"] == "needs_escalation":
        return "memory_writer"
    return "supervisor"


def route_after_memory_writer(state: OrchestraState) -> str:
    # Success clears working memory on the way out (delivery_node); an escalation deliberately
    # skips delivery — instead it goes to the approval queue — so working memory stays intact
    # in Redis for whoever picks up the escalation.
    return "delivery" if state["status"] == "complete" else "approval_queue"


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(OrchestraState)

    graph.add_node("intake", intake_node)
    graph.add_node("supervisor", SupervisorAgent())
    graph.add_node("research", ResearchAgent(), retry_policy=SPECIALIST_RETRY_POLICY)
    graph.add_node("data_analysis", DataAnalysisAgent(), retry_policy=SPECIALIST_RETRY_POLICY)
    graph.add_node("writing", WritingAgent(), retry_policy=SPECIALIST_RETRY_POLICY)
    graph.add_node("code_execution", CodeExecutionAgent(), retry_policy=SPECIALIST_RETRY_POLICY)
    graph.add_node("reviewer", ReviewerAgent())
    graph.add_node("memory_writer", MemoryWriterAgent())
    graph.add_node("approval_queue", approval_queue_node)
    graph.add_node("delivery", delivery_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            **{domain: domain for domain in SPECIALIST_DOMAINS},
            "memory_writer": "memory_writer",
            "end": END,
        },
    )

    for domain in SPECIALIST_DOMAINS:
        graph.add_edge(domain, "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"supervisor": "supervisor", "memory_writer": "memory_writer"},
    )

    graph.add_conditional_edges(
        "memory_writer",
        route_after_memory_writer,
        {"delivery": "delivery", "approval_queue": "approval_queue"},
    )
    graph.add_edge("approval_queue", END)
    graph.add_edge("delivery", END)

    return graph.compile()
