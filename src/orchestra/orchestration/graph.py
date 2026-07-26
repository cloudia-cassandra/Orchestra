"""Wires the intake / supervisor / specialist / reviewer / delivery nodes into a LangGraph
StateGraph.

Pipeline: task intake -> planning -> parallel/sequential specialist execution -> review ->
synthesis -> delivery.

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
- If the reviewer's confidence is low, or a step exhausts its attempts without approval, the
  reviewer marks it escalated and the whole run halts in "needs_escalation" rather than
  delivering a synthesis built on an unresolved step. Phase 3 (human-in-the-loop) is what turns
  that halt into an actual pause-for-a-human step instead of just stopping.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy, Send

from orchestra.agents.reviewer import ReviewerAgent
from orchestra.agents.specialists.code_execution import CodeExecutionAgent
from orchestra.agents.specialists.data_analysis import DataAnalysisAgent
from orchestra.agents.specialists.research import ResearchAgent
from orchestra.agents.specialists.writing import WritingAgent
from orchestra.agents.supervisor import SupervisorAgent
from orchestra.orchestration.state import OrchestraState
from orchestra.orchestration.waves import ready_steps

SPECIALIST_DOMAINS = ("research", "data_analysis", "writing", "code_execution")

# Retries transient specialist failures (API hiccups, rate limits) before giving up. This is
# distinct from reviewer-driven retries: this handles the node *crashing*, not the node
# succeeding with output the reviewer doesn't like.
SPECIALIST_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=1.0)


def intake_node(state: OrchestraState) -> dict:
    task = (state.get("task") or "").strip()
    if not task:
        raise ValueError("OrchestraState.task must be a non-empty string.")
    return {"status": "planning"}


def delivery_node(state: OrchestraState) -> dict:
    # Nothing to transform — this is a named seam for Phase 4 to hook final-result
    # logging/notification into, without cluttering the supervisor's synthesis logic.
    return {}


def route_after_supervisor(state: OrchestraState) -> str | list[Send]:
    if state["status"] == "complete":
        return "delivery"
    if state["status"] == "needs_escalation":
        # TODO(Phase 3): route to a human-in-the-loop escalation node instead of ending.
        return "end"

    ready = ready_steps(state)
    if not ready:
        # A well-formed DAG with steps still incomplete always has a ready step. Getting here
        # means something upstream is inconsistent — surface it rather than looping forever.
        return "end"
    return [Send(step.domain, {**state, "active_step_id": step.step_id}) for step in ready]


def route_after_reviewer(state: OrchestraState) -> str:
    if state["status"] == "needs_escalation":
        # TODO(Phase 3): route to a human-in-the-loop escalation node instead of ending.
        return "end"
    return "supervisor"


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(OrchestraState)

    graph.add_node("intake", intake_node)
    graph.add_node("supervisor", SupervisorAgent())
    graph.add_node("research", ResearchAgent(), retry_policy=SPECIALIST_RETRY_POLICY)
    graph.add_node("data_analysis", DataAnalysisAgent(), retry_policy=SPECIALIST_RETRY_POLICY)
    graph.add_node("writing", WritingAgent(), retry_policy=SPECIALIST_RETRY_POLICY)
    graph.add_node("code_execution", CodeExecutionAgent(), retry_policy=SPECIALIST_RETRY_POLICY)
    graph.add_node("reviewer", ReviewerAgent())
    graph.add_node("delivery", delivery_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {**{domain: domain for domain in SPECIALIST_DOMAINS}, "delivery": "delivery", "end": END},
    )

    for domain in SPECIALIST_DOMAINS:
        graph.add_edge(domain, "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {"supervisor": "supervisor", "end": END},
    )

    graph.add_edge("delivery", END)

    return graph.compile()
