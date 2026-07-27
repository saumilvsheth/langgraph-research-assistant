"""LangGraph workflow assembly for the research assistant."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from research_assistant.nodes import (
    generate_report_node,
    human_input_node,
    prepare_analyst_node,
    tavily_search_node,
)
from research_assistant.state import ResearchState


def build_research_graph(*, checkpointer=None):
    """
    Build and compile the research assistant LangGraph.

    Flow:
      prepare_analyst -> human_input (interrupt) -> tavily_search -> generate_report
    """
    builder = StateGraph(ResearchState)

    builder.add_node("prepare_analyst", prepare_analyst_node)
    builder.add_node("human_input", human_input_node)
    builder.add_node("tavily_search", tavily_search_node)
    builder.add_node("generate_report", generate_report_node)

    builder.add_edge(START, "prepare_analyst")
    builder.add_edge("prepare_analyst", "human_input")
    builder.add_edge("human_input", "tavily_search")
    builder.add_edge("tavily_search", "generate_report")
    builder.add_edge("generate_report", END)

    memory = checkpointer or MemorySaver()
    return builder.compile(checkpointer=memory)
