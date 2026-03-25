"""
supervisor
----------
Routing, quality gates, and error handling for the eco-material pipeline.

Public API:
    SupervisorNode      — the LangGraph node (owns all routing)
    PipelineStage       — stage identifier enum (write to state["pipeline_stage"])
    NextNode            — next-node name enum (used in Command(goto=...))
    QualityThresholds   — gate thresholds (override in tests)
    AbortReason         — pydantic schema for pipeline abort messages
"""

from .node import SupervisorNode
from .schemas import (
    AbortReason,
    NextNode,
    PipelineStage,
    QualityThresholds,
)

__all__ = [
    "SupervisorNode",
    "PipelineStage",
    "NextNode",
    "QualityThresholds",
    "AbortReason",
]
