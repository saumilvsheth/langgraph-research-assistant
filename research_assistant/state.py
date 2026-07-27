"""Graph state definition."""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages

from research_assistant.models import Analyst, CouncilEvaluation, ResearchReport
from research_assistant.usage import UsageSummary


class ResearchState(TypedDict):
    """State passed between nodes in the research assistant graph."""

    analyst: Analyst
    human_guidance: Optional[str]
    search_query: str
    search_results: list[dict]
    report: Optional[ResearchReport]
    council_evaluation: Optional[CouncilEvaluation]
    revision_count: int
    revision_feedback: Optional[str]
    human_approved: Optional[bool]
    usage: UsageSummary
    messages: Annotated[list, add_messages]
