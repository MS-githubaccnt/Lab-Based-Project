"""
explanation/schemas.py
-----------------------
Pydantic output schema for the explanation node.

Intentionally minimal — the explanation is prose, not structured data.
The schema exists to:
  1. Guarantee the LLM returned a non-empty string
  2. Keep the interface consistent with the rest of the pipeline
     (all nodes use response_model where the output goes into GraphState)
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ExplanationOutput(BaseModel):
    """
    Structured wrapper around the free-text explanation produced by
    the explanation node.
    """

    explanation: str = Field(
        description=(
            "Plain-language explanation connecting the part geometry and "
            "inferred loads to the material recommendation. "
            "Must be 2–4 sentences. Must cite specific numbers. "
            "Must not use marketing language."
        )
    )

    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Structural warnings that the engineer must be aware of. "
            "Populated when stiffness_dominated=True, buckling_risk=True, "
            "open mesh, or low inference confidence. "
            "Empty list when no caveats apply."
        )
    )

    @field_validator("explanation")
    @classmethod
    def explanation_non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("explanation must not be empty.")
        if len(stripped) < 40:
            raise ValueError(
                f"explanation is too short ({len(stripped)} chars). "
                "A meaningful explanation requires at least 40 characters."
            )
        return stripped

    @field_validator("caveats")
    @classmethod
    def caveats_non_empty_strings(cls, v: list[str]) -> list[str]:
        return [c.strip() for c in v if c.strip()]
