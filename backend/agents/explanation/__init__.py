"""
explanation
-----------
Agent node for generating grounded material recommendation explanations.

Public API:
    ExplanationNode    — the LangGraph node
    ExplanationOutput  — pydantic output schema
"""

from .node import ExplanationNode
from .schemas import ExplanationOutput

__all__ = [
    "ExplanationNode",
    "ExplanationOutput",
]
