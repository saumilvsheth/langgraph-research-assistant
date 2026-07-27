"""LangGraph node functions for the research assistant."""

import json
import os
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.types import interrupt

from research_assistant.models import (
    Analyst,
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


def prepare_analyst_node(state: ResearchState) -> dict:
    """
    Validate analyst input and build an initial search query from their profile.

    This node runs first and ensures we have a well-formed analyst persona
    before pausing for human input or hitting Tavily.
    """
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
        "usage": usage,
    }


def human_input_node(state: ResearchState) -> dict:
    """
    Pause execution and ask a human reviewer for optional research guidance.

    Uses LangGraph's interrupt() so the graph can resume later with the
    reviewer's input via Command(resume=...).
    """
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
    """
    Run Tavily search using the analyst profile and any human guidance.

    Search runs AFTER the interrupt so results stay stable on resume.
    """
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
    """Generate a structured research report from Tavily results."""
    analyst: Analyst = state["analyst"]
    search_results = state["search_results"]
    human_guidance = state.get("human_guidance")
    usage: UsageSummary = state.get("usage") or UsageSummary()
    model = _get_model_name()

    llm = _get_llm().with_structured_output(ResearchReport, include_raw=True)

    sources_text = json.dumps(search_results, indent=2)
    guidance_block = (
        f"\nHuman reviewer guidance:\n{human_guidance}\n"
        if human_guidance
        else "\nNo additional human guidance was provided.\n"
    )

    system_prompt = f"""You are {analyst.name}, a {analyst.role}.

Your background:
{analyst.description}

Write a research report from this analyst's perspective using ONLY the provided
search results. Be specific, cite sources in the sources field, and keep the tone
appropriate for a {analyst.role}. Today's date is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}."""

    user_prompt = f"""Search query used:
{state['search_query']}
{guidance_block}
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
    raw_message = result["raw"]
    record_openai_usage(usage, step="generate_report", model=model, message=raw_message)

    report.analyst_name = analyst.name
    report.analyst_role = analyst.role
    report.generated_at = datetime.now(timezone.utc).isoformat()
    report.usage = usage

    return {"report": report, "usage": usage}
