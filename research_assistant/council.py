"""Model council configuration and evaluation logic."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from research_assistant.models import (
    Analyst,
    CouncilEvaluation,
    CouncilMemberReview,
    ResearchReport,
)
from research_assistant.trace import trace
from research_assistant.usage import UsageSummary, record_openai_usage


@dataclass(frozen=True)
class CouncilMemberConfig:
    """Configuration for a single council reviewer."""

    name: str
    role: str
    model: str
    focus: str


DEFAULT_COUNCIL_MEMBERS: tuple[CouncilMemberConfig, ...] = (
    CouncilMemberConfig(
        name="Fact Checker",
        role="Factual Accuracy Reviewer",
        model=os.getenv("COUNCIL_FACT_CHECKER_MODEL", "gpt-4o-mini"),
        focus="Verify every claim is supported by the cited sources. Flag hallucinations.",
    ),
    CouncilMemberConfig(
        name="Methodology Critic",
        role="Analytical Rigor Reviewer",
        model=os.getenv("COUNCIL_METHODOLOGY_MODEL", "gpt-4o"),
        focus="Evaluate logical structure, completeness, and strength of reasoning.",
    ),
    CouncilMemberConfig(
        name="Domain Expert",
        role="Subject-Matter Reviewer",
        model=os.getenv("COUNCIL_DOMAIN_MODEL", "gpt-4o-mini"),
        focus="Assess whether the report matches the analyst persona and domain standards.",
    ),
    CouncilMemberConfig(
        name="Risk Reviewer",
        role="Bias and Risk Reviewer",
        model=os.getenv("COUNCIL_RISK_MODEL", "gpt-4o-mini"),
        focus="Identify overconfidence, missing counterarguments, bias, and blind spots.",
    ),
)


def get_council_threshold() -> float:
    trace("council", "get_council_threshold", "ENTER")
    threshold = float(os.getenv("COUNCIL_THRESHOLD", "7.0"))
    trace("council", "get_council_threshold", "EXIT", f"threshold={threshold}")
    return threshold


def get_max_revisions() -> int:
    trace("council", "get_max_revisions", "ENTER")
    max_revisions = int(os.getenv("COUNCIL_MAX_REVISIONS", "2"))
    trace("council", "get_max_revisions", "EXIT", f"max_revisions={max_revisions}")
    return max_revisions


def get_chair_model() -> str:
    trace("council", "get_chair_model", "ENTER")
    model = os.getenv("COUNCIL_CHAIR_MODEL", "gpt-4o")
    trace("council", "get_chair_model", "EXIT", f"model={model}")
    return model


def _build_llm(model: str) -> ChatOpenAI:
    trace("council", "_build_llm", "ENTER", f"model={model}")
    llm = ChatOpenAI(model=model, temperature=0.1)
    trace("council", "_build_llm", "EXIT")
    return llm


def _report_to_text(report: ResearchReport) -> str:
    trace("council", "_report_to_text", "ENTER", f"title={report.title!r}")
    text = json.dumps(
        {
            "title": report.title,
            "executive_summary": report.executive_summary,
            "key_findings": report.key_findings,
            "detailed_analysis": report.detailed_analysis,
            "recommendations": report.recommendations,
            "sources": [s.model_dump() for s in report.sources],
        },
        indent=2,
    )
    trace("council", "_report_to_text", "EXIT", f"chars={len(text)}")
    return text


def _evaluate_member_sync(
    member: CouncilMemberConfig,
    *,
    report: ResearchReport,
    analyst: Analyst,
    search_results: list[dict],
    usage: UsageSummary,
    revision_feedback: str | None,
) -> CouncilMemberReview:
    """Run a single council member evaluation (sync)."""
    trace("council", "_evaluate_member_sync", "ENTER", f"member={member.name}, model={member.model}")
    llm = _build_llm(member.model).with_structured_output(CouncilMemberReview, include_raw=True)

    feedback_block = ""
    if revision_feedback:
        feedback_block = f"\nPrevious revision feedback to consider:\n{revision_feedback}\n"

    system_prompt = f"""You are {member.name}, a {member.role} on a research review council.

Your evaluation focus: {member.focus}

Score each dimension from 1-10. Be rigorous and specific.
Recommend one of: approve, revise, reject."""

    user_prompt = f"""Review this research report for analyst {analyst.name} ({analyst.role}).

Analyst background:
{analyst.description}
{feedback_block}
Report:
{_report_to_text(report)}

Original search results (for fact checking):
{json.dumps(search_results[:5], indent=2)}

Provide your structured review as {member.name}."""

    result = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    review: CouncilMemberReview = result["parsed"]
    review.reviewer_name = member.name
    review.reviewer_role = member.role
    review.model_used = member.model

    record_openai_usage(
        usage,
        step=f"council_{member.name.lower().replace(' ', '_')}",
        model=member.model,
        message=result["raw"],
    )
    trace(
        "council",
        "_evaluate_member_sync",
        "EXIT",
        f"member={member.name}, score={review.overall_score}, rec={review.recommendation}",
    )
    return review


async def _evaluate_member_async(
    member: CouncilMemberConfig,
    **kwargs,
) -> CouncilMemberReview:
    """Async wrapper so council members can run in parallel."""
    trace("council", "_evaluate_member_async", "ENTER", f"member={member.name}")
    review = await asyncio.to_thread(_evaluate_member_sync, member, **kwargs)
    trace("council", "_evaluate_member_async", "EXIT", f"member={member.name}")
    return review


async def run_council_parallel(
    *,
    report: ResearchReport,
    analyst: Analyst,
    search_results: list[dict],
    usage: UsageSummary,
    revision_feedback: str | None = None,
    members: tuple[CouncilMemberConfig, ...] = DEFAULT_COUNCIL_MEMBERS,
) -> list[CouncilMemberReview]:
    """Run all council members in parallel."""
    member_names = [m.name for m in members]
    trace("council", "run_council_parallel", "ENTER", f"members={member_names}")
    tasks = [
        _evaluate_member_async(
            member,
            report=report,
            analyst=analyst,
            search_results=search_results,
            usage=usage,
            revision_feedback=revision_feedback,
        )
        for member in members
    ]
    reviews = list(await asyncio.gather(*tasks))
    trace("council", "run_council_parallel", "EXIT", f"reviews={len(reviews)}")
    return reviews


def synthesize_council(
    reviews: list[CouncilMemberReview],
    *,
    report: ResearchReport,
    usage: UsageSummary,
    revision_count: int,
) -> CouncilEvaluation:
    """Chair model synthesizes individual reviews into a final council evaluation."""
    trace("council", "synthesize_council", "ENTER", f"reviews={len(reviews)}, revision={revision_count}")
    chair_model = get_chair_model()
    llm = _build_llm(chair_model).with_structured_output(CouncilEvaluation, include_raw=True)

    reviews_text = json.dumps([r.model_dump() for r in reviews], indent=2)
    threshold = get_council_threshold()

    system_prompt = f"""You are the Council Chair. Synthesize reviewer opinions into one evaluation.

Approval threshold: {threshold}/10 consensus score.
Current revision attempt: {revision_count}.

Rules:
- Compute consensus_score as a weighted average of reviewer overall_score values.
- Set final_verdict to "approved" if consensus_score >= {threshold} and no reviewer recommends reject.
- Set final_verdict to "needs_revision" if consensus_score < {threshold} or majority recommends revise.
- Set final_verdict to "rejected" only if multiple reviewers recommend reject."""

    user_prompt = f"""Report title: {report.title}

Individual council reviews:
{reviews_text}

Produce the final CouncilEvaluation with member_reviews included."""

    result = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    evaluation: CouncilEvaluation = result["parsed"]
    evaluation.member_reviews = reviews
    evaluation.revision_count = revision_count
    evaluation.threshold_used = threshold

    record_openai_usage(
        usage,
        step="council_chair",
        model=chair_model,
        message=result["raw"],
    )
    trace(
        "council",
        "synthesize_council",
        "EXIT",
        f"score={evaluation.consensus_score}, verdict={evaluation.final_verdict}",
    )
    return evaluation
