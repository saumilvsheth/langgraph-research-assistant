"""Graph state definition."""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages

from research_assistant.models import Analyst, ResearchReport


class ResearchState(TypedDict):
    """State passed between nodes in the research assistant graph."""

    analyst: Analyst
    human_guidance: Optional[str]
    search_query: str
    search_results: list[dict]
    report: Optional[ResearchReport]
    messages: Annotated[list, add_messages]
