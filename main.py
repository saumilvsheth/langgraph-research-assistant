#!/usr/bin/env python3
"""
CLI runner for the LangGraph research assistant.

Demonstrates:
  1. Passing an analyst (role, name, description)
  2. Handling human-in-the-loop interrupts
  3. Tavily web search
  4. Advanced model council evaluation
  5. Structured report output with token usage and cost
"""

from __future__ import annotations

import argparse
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langgraph.types import Command

from research_assistant.graph import build_research_graph
from research_assistant.models import (
    Analyst,
    CouncilEvaluation,
    CouncilVerdictResponse,
    HumanInputResponse,
    ResearchReport,
)
from research_assistant.trace import trace
from research_assistant.usage import UsageSummary


def _print_guidance_interrupt(payload: dict[str, Any]) -> None:
    """Pretty-print the pre-search human guidance interrupt."""
    trace("cli", "_print_guidance_interrupt", "ENTER")
    print("\n" + "=" * 60)
    print("HUMAN INPUT REQUIRED — RESEARCH GUIDANCE")
    print("=" * 60)
    print(payload.get("message", ""))
    analyst = payload.get("analyst", {})
    if analyst:
        print(f"\nAnalyst: {analyst.get('name')} ({analyst.get('role')})")
        print(f"Focus: {analyst.get('description')}")
    print("\nPress Enter to continue without extra guidance,")
    print("or type additional research instructions:")
    print("-" * 60)
    trace("cli", "_print_guidance_interrupt", "EXIT")


def _print_council_interrupt(payload: dict[str, Any]) -> None:
    """Pretty-print the council verdict interrupt."""
    trace("cli", "_print_council_interrupt", "ENTER")
    print("\n" + "=" * 60)
    print("HUMAN INPUT REQUIRED — COUNCIL VERDICT")
    print("=" * 60)
    print(payload.get("message", ""))

    evaluation = payload.get("evaluation")
    if evaluation:
        if isinstance(evaluation, dict):
            evaluation = CouncilEvaluation.model_validate(evaluation)
        print(f"\nConsensus score : {evaluation.consensus_score:.1f}/10")
        print(f"Council verdict : {evaluation.final_verdict}")
        print(f"\nChair synthesis:\n{evaluation.synthesis}")
        print("\n--- Individual Reviews ---")
        for review in evaluation.member_reviews:
            print(
                f"\n{review.reviewer_name} ({review.model_used}) — "
                f"{review.overall_score}/10 → {review.recommendation}"
            )
            print(f"  Strengths : {', '.join(review.strengths[:2])}")
            print(f"  Weaknesses: {', '.join(review.weaknesses[:2])}")

    print("\nType one of: approve | revise | reject")
    print("Add notes after a colon, e.g.  revise: strengthen source citations")
    print("-" * 60)
    trace("cli", "_print_council_interrupt", "EXIT")


def _collect_guidance_response() -> HumanInputResponse:
    trace("cli", "_collect_guidance_response", "ENTER")
    try:
        user_text = input("> ").strip()
    except EOFError:
        user_text = ""
    response = HumanInputResponse(additional_guidance=user_text or None)
    trace("cli", "_collect_guidance_response", "EXIT", f"guidance={response.additional_guidance!r}")
    return response


def _collect_council_verdict() -> CouncilVerdictResponse:
    trace("cli", "_collect_council_verdict", "ENTER")
    try:
        user_text = input("> ").strip()
    except EOFError:
        user_text = "approve"

    text = user_text.lower()
    if not text:
        trace("cli", "_collect_council_verdict", "EXIT", "decision=approve (empty)")
        return CouncilVerdictResponse(decision="approve")

    if ":" in text:
        decision, notes = text.split(":", 1)
        response = CouncilVerdictResponse(decision=decision.strip(), human_notes=notes.strip() or None)
    else:
        decision = text.split()[0]
        if decision not in {"approve", "revise", "reject"}:
            decision = "approve"
        response = CouncilVerdictResponse(decision=decision)

    trace("cli", "_collect_council_verdict", "EXIT", f"decision={response.decision}")
    return response


def _format_usd(amount: float) -> str:
    if amount < 0.01:
        return f"${amount:.6f}"
    return f"${amount:.4f}"


def _print_usage(usage: UsageSummary) -> None:
    trace("cli", "_print_usage", "ENTER", f"steps={len(usage.steps)}")
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
    trace("cli", "_print_usage", "EXIT")


def _print_council_evaluation(evaluation: CouncilEvaluation) -> None:
    trace("cli", "_print_council_evaluation", "ENTER", f"score={evaluation.consensus_score}")
    print("\n" + "=" * 60)
    print("MODEL COUNCIL EVALUATION")
    print("=" * 60)
    print(f"Consensus score : {evaluation.consensus_score:.1f}/10")
    print(f"Threshold       : {evaluation.threshold_used:.1f}/10")
    print(f"Final verdict   : {evaluation.final_verdict}")
    print(f"Revision cycles : {evaluation.revision_count}")
    print(f"\nChair synthesis:\n{evaluation.synthesis}")

    if evaluation.revision_priorities:
        print("\n--- Revision Priorities ---")
        for i, item in enumerate(evaluation.revision_priorities, start=1):
            print(f"  {i}. {item}")

    print("\n--- Member Reviews ---")
    for review in evaluation.member_reviews:
        print(f"\n{review.reviewer_name} ({review.reviewer_role})")
        print(f"  Model         : {review.model_used}")
        print(f"  Overall       : {review.overall_score}/10")
        print(f"  Factual       : {review.factual_accuracy_score}/10")
        print(f"  Rigor         : {review.analytical_rigor_score}/10")
        print(f"  Domain fit    : {review.domain_fit_score}/10")
        print(f"  Recommendation: {review.recommendation}")
        print(f"  Feedback      : {review.feedback}")
    trace("cli", "_print_council_evaluation", "EXIT")


def _print_report(report: ResearchReport) -> None:
    trace("cli", "_print_report", "ENTER", f"title={report.title!r}")
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(f"Analyst : {report.analyst_name} ({report.analyst_role})")
    print(f"Title   : {report.title}")
    print(f"Generated: {report.generated_at}")
    if report.human_approved is not None:
        status = "Approved" if report.human_approved else "Rejected"
        print(f"Human verdict: {status}")

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

    if report.council_evaluation:
        _print_council_evaluation(report.council_evaluation)

    if report.usage:
        _print_usage(report.usage)
    trace("cli", "_print_report", "EXIT")


def run_research(analyst: Analyst, *, thread_id: str | None = None) -> ResearchReport:
    """Run the full research workflow, handling all interrupts along the way."""
    trace("cli", "run_research", "ENTER", f"analyst={analyst.name}, thread={thread_id}")
    graph = build_research_graph()
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    initial_input = {
        "analyst": analyst,
        "human_guidance": None,
        "search_query": "",
        "search_results": [],
        "report": None,
        "council_evaluation": None,
        "revision_count": 0,
        "revision_feedback": None,
        "human_approved": None,
        "usage": UsageSummary(),
        "messages": [],
    }

    stream_input: dict | Command = initial_input
    invoke_count = 0

    while True:
        invoke_count += 1
        trace("cli", "run_research", "INVOKE", f"graph.invoke #{invoke_count}")
        result = graph.invoke(stream_input, config=config)

        interrupts = result.get("__interrupt__")
        if not interrupts:
            trace("cli", "run_research", "BLOCK", "no interrupt — graph finished")
            break

        interrupt_payload = interrupts[0].value
        if not isinstance(interrupt_payload, dict):
            interrupt_payload = {"type": "human_guidance", "message": str(interrupt_payload)}

        interrupt_type = interrupt_payload.get("type", "human_guidance")
        trace("cli", "run_research", "INTERRUPT", f"type={interrupt_type}")

        if interrupt_type == "council_verdict":
            _print_council_interrupt(interrupt_payload)
            response = _collect_council_verdict()
            stream_input = Command(resume=response.model_dump())
        else:
            _print_guidance_interrupt(interrupt_payload)
            response = _collect_guidance_response()
            stream_input = Command(resume=response.model_dump())

    report = result.get("report")
    if report is None:
        raise RuntimeError("Graph finished without producing a report.")

    if isinstance(report, dict):
        report = ResearchReport.model_validate(report)

    usage = result.get("usage")
    if usage and report.usage is None:
        report.usage = (
            UsageSummary.model_validate(usage) if isinstance(usage, dict) else usage
        )

    if result.get("human_approved") is not None:
        report.human_approved = result["human_approved"]

    council_evaluation = result.get("council_evaluation")
    if council_evaluation and report.council_evaluation is None:
        report.council_evaluation = (
            CouncilEvaluation.model_validate(council_evaluation)
            if isinstance(council_evaluation, dict)
            else council_evaluation
        )

    trace("cli", "run_research", "EXIT", f"title={report.title!r}")
    return report


def main() -> int:
    trace("cli", "main", "ENTER")
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="LangGraph research assistant with model council evaluation"
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

    trace("cli", "main", "EXIT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
