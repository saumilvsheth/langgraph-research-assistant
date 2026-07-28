"""LangGraph workflow assembly for the research assistant."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research_assistant.nodes import (
    council_evaluate_node,
    council_human_verdict_node,
    generate_report_node,
    human_input_node,
    prepare_analyst_node,
    regenerate_report_node,
    route_after_chair,
    route_after_human_verdict,
    tavily_search_node,
)
from research_assistant.state import ResearchState
from research_assistant.trace import trace


def build_research_graph(*, checkpointer=None):
    """
    Build and compile the research assistant LangGraph.

    Flow:
      prepare_analyst -> human_input (interrupt) -> tavily_search -> generate_report
      -> council_evaluate -> route_after_chair
         -> regenerate_report (loop) OR council_human_verdict (interrupt) -> END
    """
    trace("graph", "build_research_graph", "ENTER")

    builder = StateGraph(ResearchState)

    builder.add_node("prepare_analyst", prepare_analyst_node)
    builder.add_node("human_input", human_input_node)
    builder.add_node("tavily_search", tavily_search_node)
    builder.add_node("generate_report", generate_report_node)
    builder.add_node("council_evaluate", council_evaluate_node)
    builder.add_node("regenerate_report", regenerate_report_node)
    builder.add_node("council_human_verdict", council_human_verdict_node)

    builder.add_edge(START, "prepare_analyst")
    builder.add_edge("prepare_analyst", "human_input")
    builder.add_edge("human_input", "tavily_search")
    builder.add_edge("tavily_search", "generate_report")
    builder.add_edge("generate_report", "council_evaluate")
    builder.add_edge("regenerate_report", "council_evaluate")

    builder.add_conditional_edges(
        "council_evaluate",
        route_after_chair,
        {
            "regenerate_report": "regenerate_report",
            "council_human_verdict": "council_human_verdict",
        },
    )

    builder.add_conditional_edges(
        "council_human_verdict",
        route_after_human_verdict,
        {
            "regenerate_report": "regenerate_report",
            "__end__": END,
        },
    )

    memory = checkpointer or MemorySaver()
    graph = builder.compile(checkpointer=memory)

    trace("graph", "build_research_graph", "EXIT", "graph compiled")
    return graph
