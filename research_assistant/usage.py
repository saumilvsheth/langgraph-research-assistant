"""Token usage and cost tracking for LLM and Tavily calls."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from research_assistant.trace import trace

# USD per 1M tokens (input, output). Override via env if pricing changes.
DEFAULT_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-3.5-turbo": (0.50, 1.50),
}

TAVILY_CREDIT_COST_USD = float(os.getenv("TAVILY_CREDIT_COST_USD", "0.008"))


class StepUsage(BaseModel):
    """Usage for a single workflow step."""

    step: str = Field(description="Workflow step name")
    provider: str = Field(description="Service provider, e.g. openai or tavily")
    model: str = Field(description="Model name or Tavily operation")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    tavily_credits: Optional[int] = None


class UsageSummary(BaseModel):
    """Aggregated token usage and estimated cost for a research run."""

    steps: list[StepUsage] = Field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_llm_cost_usd: float = 0.0
    total_tavily_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    def add_step(self, step: StepUsage) -> None:
        """Append a step and refresh aggregate totals."""
        trace("usage", "UsageSummary.add_step", "ENTER", f"step={step.step}")
        self.steps.append(step)
        self._recompute_totals()
        trace("usage", "UsageSummary.add_step", "EXIT", f"total_steps={len(self.steps)}")

    def _recompute_totals(self) -> None:
        trace("usage", "UsageSummary._recompute_totals", "ENTER")
        self.total_prompt_tokens = sum(s.prompt_tokens for s in self.steps)
        self.total_completion_tokens = sum(s.completion_tokens for s in self.steps)
        self.total_tokens = sum(s.total_tokens for s in self.steps)
        self.total_llm_cost_usd = sum(
            s.total_cost_usd for s in self.steps if s.provider == "openai"
        )
        self.total_tavily_cost_usd = sum(
            s.total_cost_usd for s in self.steps if s.provider == "tavily"
        )
        self.total_cost_usd = self.total_llm_cost_usd + self.total_tavily_cost_usd
        trace(
            "usage",
            "UsageSummary._recompute_totals",
            "EXIT",
            f"total_cost=${self.total_cost_usd:.6f}",
        )


def _get_model_pricing(model: str) -> tuple[float, float]:
    """Return (input_per_1m, output_per_1m) for a model."""
    trace("usage", "_get_model_pricing", "ENTER", f"model={model}")
    if model in DEFAULT_MODEL_PRICING:
        pricing = DEFAULT_MODEL_PRICING[model]
    else:
        pricing = None
        for key, value in DEFAULT_MODEL_PRICING.items():
            if model.startswith(key):
                pricing = value
                break
        if pricing is None:
            pricing = (
                float(os.getenv("OPENAI_DEFAULT_INPUT_COST_PER_1M", "0.15")),
                float(os.getenv("OPENAI_DEFAULT_OUTPUT_COST_PER_1M", "0.60")),
            )
    trace("usage", "_get_model_pricing", "EXIT", f"pricing={pricing}")
    return pricing


def _extract_token_counts(message: AIMessage) -> tuple[int, int, int]:
    """Read prompt/completion/total token counts from an AIMessage."""
    trace("usage", "_extract_token_counts", "ENTER")
    usage_meta = getattr(message, "usage_metadata", None) or {}
    if usage_meta:
        prompt = int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens") or 0)
        completion = int(
            usage_meta.get("output_tokens") or usage_meta.get("completion_tokens") or 0
        )
        total = int(usage_meta.get("total_tokens") or (prompt + completion))
        trace("usage", "_extract_token_counts", "EXIT", f"tokens={total}")
        return prompt, completion, total

    token_usage = (getattr(message, "response_metadata", None) or {}).get("token_usage", {})
    prompt = int(token_usage.get("prompt_tokens") or 0)
    completion = int(token_usage.get("completion_tokens") or 0)
    total = int(token_usage.get("total_tokens") or (prompt + completion))
    trace("usage", "_extract_token_counts", "EXIT", f"tokens={total}")
    return prompt, completion, total


def record_openai_usage(
    summary: UsageSummary,
    *,
    step: str,
    model: str,
    message: AIMessage,
) -> StepUsage:
    """Record token usage from an OpenAI chat response."""
    trace("usage", "record_openai_usage", "ENTER", f"step={step}, model={model}")
    prompt_tokens, completion_tokens, total_tokens = _extract_token_counts(message)
    input_rate, output_rate = _get_model_pricing(model)

    input_cost = (prompt_tokens / 1_000_000) * input_rate
    output_cost = (completion_tokens / 1_000_000) * output_rate

    usage = StepUsage(
        step=step,
        provider="openai",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        input_cost_usd=round(input_cost, 6),
        output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(input_cost + output_cost, 6),
    )
    summary.add_step(usage)
    trace("usage", "record_openai_usage", "EXIT", f"tokens={total_tokens}")
    return usage


def record_tavily_usage(
    summary: UsageSummary,
    *,
    step: str,
    search_depth: str,
    raw_results: dict[str, Any],
) -> StepUsage:
    """Record Tavily credit usage when returned by the API."""
    trace("usage", "record_tavily_usage", "ENTER", f"step={step}, depth={search_depth}")
    usage_info = raw_results.get("usage") if isinstance(raw_results, dict) else None
    credits = None
    if isinstance(usage_info, dict):
        credits = usage_info.get("credits") or usage_info.get("total_credits")
    elif isinstance(usage_info, (int, float)):
        credits = int(usage_info)

    if credits is None:
        credits = 2 if search_depth == "advanced" else 1

    total_cost = credits * TAVILY_CREDIT_COST_USD
    usage = StepUsage(
        step=step,
        provider="tavily",
        model=f"tavily-search ({search_depth})",
        total_cost_usd=round(total_cost, 6),
        tavily_credits=int(credits),
    )
    summary.add_step(usage)
    trace("usage", "record_tavily_usage", "EXIT", f"credits={credits}")
    return usage
