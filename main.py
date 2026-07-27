#!/usr/bin/env python3
"""
CLI runner for the LangGraph research assistant.

Demonstrates:
  1. Passing an analyst (role, name, description)
  2. Handling human-in-the-loop interrupts
  3. Tavily web search
  4. Structured report output with token usage and cost
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langgraph.types import Command

from research_assistant.graph import build_research_graph
from research_assistant.models import Analyst, HumanInputResponse, ResearchReport
from research_assistant.usage import UsageSummary


def _print_interrupt(payload: dict[str, Any]) -> None:
    """Pretty-print the interrupt payload shown to the human reviewer."""
    print("\n" + "=" * 60)
    print("HUMAN INPUT REQUIRED")
    print("=" * 60)
    print(payload.get("message", ""))
    analyst = payload.get("analyst", {})
    if analyst:
        print(f"\nAnalyst: {analyst.get('name')} ({analyst.get('role')})")
        print(f"Focus: {analyst.get('description')}")
    print("\nPress Enter to continue without extra guidance,")
    print("or type additional research instructions:")
    print("-" * 60)


def _collect_human_response() -> HumanInputResponse:
    """Read optional guidance from stdin after an interrupt."""
    try:
        user_text = input("> ").strip()
    except EOFError:
        user_text = ""
    return HumanInputResponse(additional_guidance=user_text or None)


def _format_usd(amount: float) -> str:
    if amount < 0.01:
        return f"${amount:.6f}"
    return f"${amount:.4f}"


def _print_usage(usage: UsageSummary) -> None:
    """Display token usage and estimated cost."""
    print("\n" + "=" * 60)
    print("TOKEN USAGE & COST")
    print("=" * 60)

    for step in usage.steps:
        print(f"\n{step.step} ({step.provider} / {step.model})")
        if step.provider == "openai":
            print(f"  Prompt tokens     : {step.prompt_tokens:,}")
            print(f"  Completion tokens : {step.completion_tokens:,}")
            print(f"  Total tokens      : {step.total_tokens:,}")
            print(f"  Input cost        : {_format_usd(step.input_cost_usd)}")
            print(f"  Output cost       : {_format_usd(step.output_cost_usd)}")
        elif step.tavily_credits is not None:
            print(f"  Tavily credits    : {step.tavily_credits}")
        print(f"  Step cost         : {_format_usd(step.total_cost_usd)}")

    print("\n--- Totals ---")
    print(f"  LLM prompt tokens     : {usage.total_prompt_tokens:,}")
    print(f"  LLM completion tokens : {usage.total_completion_tokens:,}")
    print(f"  LLM total tokens      : {usage.total_tokens:,}")
    print(f"  LLM cost              : {_format_usd(usage.total_llm_cost_usd)}")
    print(f"  Tavily cost           : {_format_usd(usage.total_tavily_cost_usd)}")
    print(f"  Estimated total cost  : {_format_usd(usage.total_cost_usd)}")


def _print_report(report: ResearchReport) -> None:
    """Display the structured report in a readable format."""
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(f"Analyst : {report.analyst_name} ({report.analyst_role})")
    print(f"Title   : {report.title}")
    print(f"Generated: {report.generated_at}")
    print("\n--- Executive Summary ---")
    print(report.executive_summary)
    print("\n--- Key Findings ---")
    for i, finding in enumerate(report.key_findings, start=1):
        print(f"  {i}. {finding}")
    print("\n--- Detailed Analysis ---")
    print(report.detailed_analysis)
    print("\n--- Recommendations ---")
    for i, rec in enumerate(report.recommendations, start=1):
        print(f"  {i}. {rec}")
    print("\n--- Sources ---")
    for source in report.sources:
        print(f"  • {source.title}")
        print(f"    {source.url}")

    if report.usage:
        _print_usage(report.usage)


def run_research(analyst: Analyst, *, thread_id: str | None = None) -> ResearchReport:
    """
    Run the full research workflow, handling interrupts along the way.

    Returns the final structured ResearchReport.
    """
    graph = build_research_graph()
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    initial_input = {
        "analyst": analyst,
        "human_guidance": None,
        "search_query": "",
        "search_results": [],
        "report": None,
        "usage": UsageSummary(),
        "messages": [],
    }

    stream_input: dict | Command = initial_input

    while True:
        result = graph.invoke(stream_input, config=config)

        interrupts = result.get("__interrupt__")
        if not interrupts:
            break

        interrupt_payload = interrupts[0].value
        _print_interrupt(interrupt_payload)
        human_response = _collect_human_response()
        stream_input = Command(resume=human_response.model_dump())

    report = result.get("report")
    if report is None:
        raise RuntimeError("Graph finished without producing a report.")

    if isinstance(report, dict):
        report = ResearchReport.model_validate(report)

    usage = result.get("usage")
    if usage and report.usage is None:
        if isinstance(usage, dict):
            usage = UsageSummary.model_validate(usage)
        report.usage = usage

    return report


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="LangGraph research assistant with human-in-the-loop interrupts"
    )
    parser.add_argument("--name", required=True, help="Analyst name")
    parser.add_argument("--role", required=True, help="Analyst role")
    parser.add_argument(
        "--description",
        required=True,
        help="Analyst background and research focus",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final report as JSON instead of formatted text",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the report JSON",
    )
    args = parser.parse_args()

    analyst = Analyst(
        name=args.name,
        role=args.role,
        description=args.description,
    )

    print(f"Starting research for {analyst.name} ({analyst.role})...")
    report = run_research(analyst)

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"\nReport saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
