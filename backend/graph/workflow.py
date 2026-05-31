from langgraph.graph import StateGraph, END
from loguru import logger

from backend.graph.state import ResearchState
from backend.agents.planner_agent import run_planner
from backend.agents.retrieval_agent import run_retrieval
from backend.agents.reflection_agent import run_reflection
from backend.agents.critic_agent import run_critic
from backend.agents.citation_agent import run_citation
from backend.agents.report_agent import run_report


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_reflection(state: ResearchState) -> str:
    """
    If reflection failed AND we haven't hit max iterations, loop back to retrieval.
    Otherwise proceed to critic.
    """
    if not state.get("reflection_passed", True) and state.get("reflection_iterations", 0) < 2:
        logger.info("[Workflow] Reflection failed — looping back to retrieval")
        return "retrieval"
    return "critic"


def route_after_critic(state: ResearchState) -> str:
    """
    If critic rejected the evidence, go straight to report with a failure message.
    Otherwise proceed to citation.
    """
    if not state.get("critic_passed", True):
        logger.info("[Workflow] Critic rejected — routing to report with failure")
        return "report"
    return "citation"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Register all agent nodes
    graph.add_node("planner", run_planner)
    graph.add_node("retrieval", run_retrieval)
    graph.add_node("reflection", run_reflection)
    graph.add_node("critic", run_critic)
    graph.add_node("citation", run_citation)
    graph.add_node("report", run_report)

    # Entry point
    graph.set_entry_point("planner")

    # Linear edges
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "reflection")

    # Conditional: reflection can loop back to retrieval
    graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "retrieval": "retrieval",
            "critic": "critic",
        },
    )

    # Conditional: critic can skip citation and go straight to report
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "citation": "citation",
            "report": "report",
        },
    )

    # Citation always goes to report
    graph.add_edge("citation", "report")

    # Report is the terminal node
    graph.add_edge("report", END)

    return graph.compile()


# Compile once at module level — reused for every query
compiled_workflow = build_workflow()
