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


class CouncilMemberReview(BaseModel):
    """Individual review from one council member."""

    reviewer_name: str = Field(description="Name of the council reviewer")
    reviewer_role: str = Field(description="Reviewer's specialty role")
    model_used: str = Field(default="", description="LLM model used for this review")
    overall_score: float = Field(ge=1, le=10, description="Overall quality score from 1-10")
    factual_accuracy_score: float = Field(ge=1, le=10, description="Factual accuracy score")
    analytical_rigor_score: float = Field(ge=1, le=10, description="Analytical rigor score")
    domain_fit_score: float = Field(ge=1, le=10, description="Fit for analyst role/domain")
    strengths: list[str] = Field(description="Specific strengths in the report")
    weaknesses: list[str] = Field(description="Specific weaknesses or gaps")
    recommendation: str = Field(description="One of: approve, revise, reject")
    feedback: str = Field(description="Detailed feedback for the report author")


class CouncilEvaluation(BaseModel):
    """Synthesized evaluation from the full model council."""

    member_reviews: list[CouncilMemberReview] = Field(description="Individual member reviews")
    consensus_score: float = Field(ge=1, le=10, description="Weighted average score across reviewers")
    final_verdict: str = Field(description="One of: approved, needs_revision, rejected")
    synthesis: str = Field(description="Chair's summary of council deliberation")
    revision_priorities: list[str] = Field(
        default_factory=list,
        description="Top issues to fix if revision is needed",
    )
    revision_count: int = Field(default=0, description="How many revision cycles have occurred")
    threshold_used: float = Field(default=7.0, description="Score threshold used for approval")


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
    council_evaluation: Optional[CouncilEvaluation] = Field(
        default=None,
        description="Model council evaluation of this report",
    )
    human_approved: Optional[bool] = Field(
        default=None,
        description="Whether a human reviewer approved the final report",
    )


class HumanInputRequest(BaseModel):
    """Payload surfaced to the user during the human-in-the-loop interrupt."""

    type: str = "human_guidance"
    message: str
    analyst: Analyst
    optional: bool = True


class HumanInputResponse(BaseModel):
    """Response collected from the user after an interrupt."""

    additional_guidance: Optional[str] = Field(
        default=None,
        description="Optional extra research guidance from the human reviewer",
    )


class CouncilVerdictRequest(BaseModel):
    """Payload for the council verdict human-in-the-loop interrupt."""

    type: str = "council_verdict"
    message: str
    evaluation: CouncilEvaluation
    report_title: str
    consensus_score: float
    final_verdict: str


class CouncilVerdictResponse(BaseModel):
    """Human decision on the council-evaluated report."""

    decision: str = Field(description="One of: approve, revise, reject")
    human_notes: Optional[str] = Field(
        default=None,
        description="Optional notes explaining the human decision",
    )
