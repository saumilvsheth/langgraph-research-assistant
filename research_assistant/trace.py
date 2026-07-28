"""Lightweight execution tracing for following function and block flow."""

from __future__ import annotations


def trace(block: str, function: str, phase: str = "ENTER", detail: str = "") -> None:
    """
    Print a trace line showing which block and function is running.

    Example output:
        >>> [nodes] prepare_analyst_node — ENTER
        >>> [routing] route_after_chair — ROUTE | next=regenerate_report
    """
    message = f">>> [{block}] {function} — {phase}"
    if detail:
        message = f"{message} | {detail}"
    print(message, flush=True)
