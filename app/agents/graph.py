"""
LangGraph identity-verification pipeline.

Graph topology (linear):

    START → capture → liveness → recognition → decision → END

All state is carried in a plain dict conforming to AgentState.
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import node_capture, node_liveness, node_recognition, node_decision
from app.models.schemas import AgentState

logger = logging.getLogger(__name__)


def _dict_reducer(a: dict, b: dict) -> dict:
    """Simple last-write-wins merge for the state dict."""
    merged = {**a}
    merged.update(b)
    return merged


def build_verification_graph():
    """
    Build and compile the LangGraph verification workflow.
    Returns a compiled graph ready for `.invoke()`.
    """
    # Use dict as the state type; nodes receive and return plain dicts
    builder: StateGraph = StateGraph(dict)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("capture", node_capture)
    builder.add_node("liveness", node_liveness)
    builder.add_node("recognition", node_recognition)
    builder.add_node("decision", node_decision)

    # ── Define edges (linear pipeline) ───────────────────────────────────
    builder.add_edge(START, "capture")
    builder.add_edge("capture", "liveness")
    builder.add_edge("liveness", "recognition")
    builder.add_edge("recognition", "decision")
    builder.add_edge("decision", END)

    return builder.compile()


# Module-level singleton — compiled once on import
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_verification_graph()
        logger.info("Verification graph compiled.")
    return _graph


def run_verification(initial_state: dict[str, Any]) -> AgentState:
    """
    Execute the full verification pipeline and return the final AgentState.
    """
    graph = get_graph()
    # Ensure all AgentState defaults are present
    base = AgentState().model_dump()
    base.update(initial_state)

    try:
        final = graph.invoke(base)
        return AgentState(**final)
    except Exception as exc:
        logger.exception("Graph execution error: %s", exc)
        error_state = AgentState(**base)
        error_state.final_message = f"Internal pipeline error: {exc}"
        from app.models.schemas import VerificationStatus
        error_state.final_status = VerificationStatus.REJECTED
        return error_state
