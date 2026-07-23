"""Wires the supervisor / specialist / reviewer nodes into a LangGraph StateGraph."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orchestra.agents.reviewer import ReviewerAgent
from orchestra.agents.specialists.code_execution import CodeExecutionAgent
from orchestra.agents.specialists.data_analysis import DataAnalysisAgent
from orchestra.agents.specialists.research import ResearchAgent
from orchestra.agents.specialists.writing import WritingAgent
from orchestra.agents.supervisor import SupervisorAgent
from orchestra.orchestration.state import OrchestraState

SPECIALIST_DOMAINS = ("research", "data_analysis", "writing", "code_execution")


def route_after_supervisor(state: OrchestraState) -> str:
    if state["status"] == "complete":
        return "end"
    return state["plan"].steps[state["current_step_index"]].domain


def route_after_review(state: OrchestraState) -> str:
    if state["status"] == "retrying":
        # Same step, same domain — send it back to the specialist that produced it.
        return state["plan"].steps[state["current_step_index"]].domain
    if state["status"] == "needs_escalation":
        # TODO(Phase 3): route to the human-in-the-loop escalation node instead of ending.
        return "end"
    return "supervisor"


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(OrchestraState)

    graph.add_node("supervisor", SupervisorAgent())
    graph.add_node("research", ResearchAgent())
    graph.add_node("data_analysis", DataAnalysisAgent())
    graph.add_node("writing", WritingAgent())
    graph.add_node("code_execution", CodeExecutionAgent())
    graph.add_node("reviewer", ReviewerAgent())

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {**{domain: domain for domain in SPECIALIST_DOMAINS}, "end": END},
    )

    for domain in SPECIALIST_DOMAINS:
        graph.add_edge(domain, "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {**{domain: domain for domain in SPECIALIST_DOMAINS}, "supervisor": "supervisor", "end": END},
    )

    return graph.compile()
