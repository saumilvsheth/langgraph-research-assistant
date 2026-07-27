"""LangGraph node functions for the research assistant."""

import asyncio
import json
import os
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.types import interrupt

from research_assistant.council import (
    get_council_threshold,
    get_max_revisions,
    run_council_parallel,
    synthesize_council,
)
from research_assistant.models import (
    Analyst,
    CouncilVerdictRequest,
    CouncilVerdictResponse,
    HumanInputRequest,
    HumanInputResponse,
    ResearchReport,
)
from research_assistant.state import ResearchState
from research_assistant.usage import UsageSummary, record_openai_usage, record_tavily_usage


def _get_llm() -> ChatOpenAI:
    """Create the chat model used for query generation and report writing."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0.2)


def _get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _get_tavily_search() -> TavilySearch:
    """Create the Tavily search tool."""
    return TavilySearch(
        max_results=int(os.getenv("TAVILY_MAX_RESULTS", "5")),
        search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "advanced"),
        include_answer=True,
        include_usage=True,
    )


def _get_search_depth() -> str:
    return os.getenv("TAVILY_SEARCH_DEPTH", "advanced")


def _build_report(
    *,
    analyst: Analyst,
    search_query: str,
    search_results: list[dict],
    human_guidance: str | None,
    revision_feedback: str | None,
    usage: UsageSummary,
    step_prefix: str = "generate_report",
) -> ResearchReport:
    """Shared report generation logic used for initial draft and revisions."""
    model = _get_model_name()
    llm = _get_llm().with_structured_output(ResearchReport, include_raw=True)

    sources_text = json.dumps(search_results, indent=2)
    guidance_block = (
        f"\nHuman reviewer guidance:\n{human_guidance}\n"
        if human_guidance
        else "\nNo additional human guidance was provided.\n"
    )
    revision_block = (
        f"\nCouncil revision feedback (address these issues):\n{revision_feedback}\n"
        if revision_feedback
        else ""
    )

    system_prompt = f"""You are {analyst.name}, a {analyst.role}.

Your background:
{analyst.description}

Write a research report from this analyst's perspective using ONLY the provided
search results. Be specific, cite sources in the sources field, and keep the tone
appropriate for a {analyst.role}. Today's date is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}."""

    user_prompt = f"""Search query used:
{search_query}
{guidance_block}{revision_block}
Search results:
{sources_text}

Create a structured research report for analyst '{analyst.name}' with role '{analyst.role}'."""

    result = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    report: ResearchReport = result["parsed"]
    record_openai_usage(usage, step=step_prefix, model=model, message=result["raw"])

    report.analyst_name = analyst.name
    report.analyst_role = analyst.role
    report.generated_at = datetime.now(timezone.utc).isoformat()
    report.usage = usage
    return report


def prepare_analyst_node(state: ResearchState) -> dict:
    """Validate analyst input and build an initial search query from their profile."""
    analyst: Analyst = state["analyst"]
    usage: UsageSummary = state.get("usage") or UsageSummary()

    llm = _get_llm()
    model = _get_model_name()
    prompt = f"""You are helping {analyst.name}, a {analyst.role}.

Analyst background:
{analyst.description}

Write one concise web search query (max 20 words) that this analyst would use
to start their research. Return ONLY the query text, no quotes or explanation."""

    response = llm.invoke([HumanMessage(content=prompt)])
    record_openai_usage(usage, step="prepare_analyst", model=model, message=response)

    search_query = response.content.strip().strip('"').strip("'")

    return {
        "search_query": search_query,
        "human_guidance": state.get("human_guidance"),
        "search_results": [],
        "report": None,
        "council_evaluation": None,
        "revision_count": state.get("revision_count", 0),
        "revision_feedback": state.get("revision_feedback"),
        "human_approved": None,
        "usage": usage,
    }


def human_input_node(state: ResearchState) -> dict:
    """Pause for optional human research guidance before search."""
    analyst: Analyst = state["analyst"]

    request = HumanInputRequest(
        message=(
            f"Research is being prepared for {analyst.name} ({analyst.role}). "
            f"Draft search query: '{state['search_query']}'. "
            "Provide additional guidance, or leave blank to continue."
        ),
        analyst=analyst,
        optional=True,
    )

    human_response = interrupt(request.model_dump())

    if isinstance(human_response, dict):
        parsed = HumanInputResponse.model_validate(human_response)
    else:
        parsed = HumanInputResponse(additional_guidance=str(human_response) or None)

    guidance = (parsed.additional_guidance or "").strip() or None
    return {"human_guidance": guidance}


def tavily_search_node(state: ResearchState) -> dict:
    """Run Tavily search using the analyst profile and any human guidance."""
    analyst: Analyst = state["analyst"]
    search_query = state["search_query"]
    human_guidance = state.get("human_guidance")
    usage: UsageSummary = state.get("usage") or UsageSummary()
    search_depth = _get_search_depth()

    if human_guidance:
        search_query = (
            f"{search_query}. Context from reviewer: {human_guidance}. "
            f"Research focus for {analyst.role}: {analyst.description[:200]}"
        )

    tavily = _get_tavily_search()
    raw_results = tavily.invoke({"query": search_query})

    if isinstance(raw_results, dict):
        record_tavily_usage(
            usage,
            step="tavily_search",
            search_depth=search_depth,
            raw_results=raw_results,
        )
        results = raw_results.get("results", [])
    else:
        results = []

    return {"search_results": results, "search_query": search_query, "usage": usage}


def generate_report_node(state: ResearchState) -> dict:
    """Generate the initial structured research report from Tavily results."""
    report = _build_report(
        analyst=state["analyst"],
        search_query=state["search_query"],
        search_results=state["search_results"],
        human_guidance=state.get("human_guidance"),
        revision_feedback=None,
        usage=state.get("usage") or UsageSummary(),
        step_prefix="generate_report",
    )
    return {"report": report, "usage": report.usage}


def regenerate_report_node(state: ResearchState) -> dict:
    """Regenerate the report using council or human revision feedback."""
    revision_count = state.get("revision_count", 0) + 1
    evaluation = state.get("council_evaluation")

    feedback_parts = []
    if evaluation:
        feedback_parts.append(evaluation.synthesis)
        feedback_parts.extend(evaluation.revision_priorities)
    if state.get("revision_feedback"):
        feedback_parts.append(state["revision_feedback"])

    combined_feedback = "\n".join(f"- {item}" for item in feedback_parts if item)

    report = _build_report(
        analyst=state["analyst"],
        search_query=state["search_query"],
        search_results=state["search_results"],
        human_guidance=state.get("human_guidance"),
        revision_feedback=combined_feedback or None,
        usage=state.get("usage") or UsageSummary(),
        step_prefix=f"regenerate_report_{revision_count}",
    )

    return {
        "report": report,
        "revision_count": revision_count,
        "revision_feedback": combined_feedback or None,
        "council_evaluation": None,
        "usage": report.usage,
    }


def council_evaluate_node(state: ResearchState) -> dict:
    """Run parallel council member evaluations on the current report."""
    report = state["report"]
    if report is None:
        raise RuntimeError("Council evaluation requires a report.")

    usage: UsageSummary = state.get("usage") or UsageSummary()

    reviews = asyncio.run(
        run_council_parallel(
            report=report,
            analyst=state["analyst"],
            search_results=state["search_results"],
            usage=usage,
            revision_feedback=state.get("revision_feedback"),
        )
    )

    evaluation = synthesize_council(
        reviews,
        report=report,
        usage=usage,
        revision_count=state.get("revision_count", 0),
    )

    report.council_evaluation = evaluation
    return {"council_evaluation": evaluation, "report": report, "usage": usage}


def council_human_verdict_node(state: ResearchState) -> dict:
    """Pause for human approval, rejection, or revision request after council review."""
    report = state["report"]
    evaluation = state["council_evaluation"]
    if report is None or evaluation is None:
        raise RuntimeError("Council verdict requires a report and evaluation.")

    request = CouncilVerdictRequest(
        message=(
            f"The model council has reviewed '{report.title}'. "
            f"Consensus score: {evaluation.consensus_score:.1f}/10. "
            f"Verdict: {evaluation.final_verdict}. "
            "Type 'approve', 'revise', or 'reject' (optionally add notes after a colon)."
        ),
        evaluation=evaluation,
        report_title=report.title,
        consensus_score=evaluation.consensus_score,
        final_verdict=evaluation.final_verdict,
    )

    human_response = interrupt(request.model_dump())

    if isinstance(human_response, dict):
        parsed = CouncilVerdictResponse.model_validate(human_response)
    else:
        parsed = _parse_verdict_text(str(human_response))

    updates: dict = {"human_approved": parsed.decision == "approve"}

    if parsed.decision == "revise":
        updates["human_approved"] = None
        updates["revision_feedback"] = (
            parsed.human_notes
            or evaluation.synthesis
            or "\n".join(evaluation.revision_priorities)
        )
    elif parsed.decision == "reject":
        updates["human_approved"] = False

    return updates


def _parse_verdict_text(text: str) -> CouncilVerdictResponse:
    """Parse simple CLI verdict input like 'approve' or 'revise: fix sources'."""
    text = text.strip().lower()
    if not text:
        return CouncilVerdictResponse(decision="approve")

    if ":" in text:
        decision, notes = text.split(":", 1)
        decision = decision.strip()
        notes = notes.strip() or None
    else:
        decision = text.split()[0]
        notes = text[len(decision) :].strip() or None

    if decision not in {"approve", "revise", "reject"}:
        decision = "approve"

    return CouncilVerdictResponse(decision=decision, human_notes=notes)


def route_after_chair(state: ResearchState) -> str:
    """Auto-regenerate if council score is below threshold and revisions remain."""
    evaluation = state.get("council_evaluation")
    if evaluation is None:
        return "council_human_verdict"

    below_threshold = evaluation.consensus_score < get_council_threshold()
    can_revise = state.get("revision_count", 0) < get_max_revisions()
    needs_revision = evaluation.final_verdict in {"needs_revision", "rejected"}

    if (below_threshold or needs_revision) and can_revise:
        return "regenerate_report"

    return "council_human_verdict"


def route_after_human_verdict(state: ResearchState) -> str:
    """Route based on the human's council verdict decision."""
    if state.get("human_approved") is True:
        return "__end__"

    evaluation = state.get("council_evaluation")
    can_revise = state.get("revision_count", 0) < get_max_revisions()

    if state.get("revision_feedback") and can_revise:
        return "regenerate_report"

    return "__end__"
