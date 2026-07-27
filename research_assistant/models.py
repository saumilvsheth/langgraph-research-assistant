"""Data models for the research assistant."""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from research_assistant.usage import UsageSummary


class Analyst(BaseModel):
    """Analyst persona that drives research focus and report tone."""

    role: str = Field(description="The analyst's professional role (e.g. 'Equity Research Analyst')")
    name: str = Field(description="The analyst's name")
    description: str = Field(
        description="Background, expertise, and research focus areas for this analyst"
    )


class Source(BaseModel):
    """A web source cited in the research report."""

    title: str = Field(description="Title of the source article or page")
    url: str = Field(description="URL of the source")
    snippet: str = Field(description="Relevant excerpt from the source")


class ResearchReport(BaseModel):
    """Structured research report generated for a specific analyst."""

    analyst_name: str = Field(description="Name of the analyst this report is for")
    analyst_role: str = Field(description="Role of the analyst")
    title: str = Field(description="Report title summarizing the research topic")
    executive_summary: str = Field(description="High-level summary of findings (2-3 paragraphs)")
    key_findings: list[str] = Field(description="Bullet-point list of the most important findings")
    detailed_analysis: str = Field(description="In-depth analysis written from the analyst's perspective")
    recommendations: list[str] = Field(description="Actionable recommendations based on the research")
    sources: list[Source] = Field(description="Sources used to compile the report")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when the report was generated",
    )
    usage: Optional[UsageSummary] = Field(
        default=None,
        description="Token usage and estimated cost for this research run",
    )


class HumanInputRequest(BaseModel):
    """Payload surfaced to the user during the human-in-the-loop interrupt."""

    message: str
    analyst: Analyst
    optional: bool = True


class HumanInputResponse(BaseModel):
    """Response collected from the user after an interrupt."""

    additional_guidance: Optional[str] = Field(
        default=None,
        description="Optional extra research guidance from the human reviewer",
    )
