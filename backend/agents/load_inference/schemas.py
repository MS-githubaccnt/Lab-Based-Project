from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


class LoadType(str, Enum):
    STATIC    = "static"
    CYCLIC    = "cyclic"
    IMPACT    = "impact"
    THERMAL   = "thermal"


class StressMode(str, Enum):
    BENDING     = "bending"
    TENSION     = "tension"
    COMPRESSION = "compression"
    TORSION     = "torsion"
    NONE        = "none"   # purely thermal or cosmetic parts


class DeflectionCategory(str, Enum):
    PRECISION  = "precision"   # instruments, optical — L/500
    STRUCTURAL = "structural"  # civil frames         — L/300
    MECHANICAL = "mechanical"  # machine parts        — L/100  (default)
    FLEXIBLE   = "flexible"    # housings, covers     — L/30


# ---------------------------------------------------------------------------
# LoadEstimate
# ---------------------------------------------------------------------------

class LoadEstimate(BaseModel):
    """
    Structured load profile inferred by the load_inference LLM node.

    All fields are required except clarification_question, which is only
    populated when confidence < CONFIDENCE_THRESHOLD.
    """

    # ── Load character ────────────────────────────────────────────────────
    load_type: LoadType = Field(
        description=(
            "Nature of the loading regime. "
            "'static': constant force (clamp, support bracket). "
            "'cyclic': repeated loading (pedal crank, connecting rod). "
            "'impact': sudden shock (bumper, protective housing). "
            "'thermal': temperature-driven expansion/contraction only."
        )
    )

    primary_stress_mode: StressMode = Field(
        description=(
            "Dominant stress mode at the critical section. "
            "'bending': off-axis or cantilever loads (most structural parts). "
            "'tension': in-line pull (cables, tensile links). "
            "'compression': in-line push (columns, press platens). "
            "'torsion': twisting (shafts, drive rods). "
            "'none': purely thermal or cosmetic — no mechanical stress."
        )
    )

    # ── Magnitude ─────────────────────────────────────────────────────────
    magnitude_range_N: Tuple[float, float] = Field(
        description=(
            "Expected force range in Newtons [min_N, max_N]. "
            "Use the typical operating range for the object class — "
            "not ultimate failure loads. "
            "Example: bicycle pedal force -> [200, 1500]. "
            "Set both to 0 for purely thermal or decorative parts."
        )
    )

    # ── Safety factor ─────────────────────────────────────────────────────
    safety_factor: float = Field(
        description=(
            "Appropriate safety factor for this object class and application. "
            "Guidelines: "
            "aerospace: 1.5, automotive: 2.0–2.5, "
            "industrial machinery: 2.5–3.5, consumer products: 3.0–4.0, "
            "life-critical structures: 4.0+. "
            "Use higher values when load estimates are uncertain."
        )
    )

    # ── Temperature ───────────────────────────────────────────────────────
    operating_temp_C: Tuple[float, float] = Field(
        description=(
            "Expected operating temperature range [min_C, max_C]. "
            "Use the environment the object will operate in, not just ambient. "
            "Example: engine bracket -> [-20, 150]. "
            "Default ambient: [-10, 60]."
        )
    )

    # ── Fatigue ───────────────────────────────────────────────────────────
    is_fatigue_critical: bool = Field(
        description=(
            "True if the part will experience >10,000 load cycles in its "
            "service life and failure would be safety-relevant. "
            "True examples: crank arms, suspension links, pressure vessels. "
            "False examples: enclosures, brackets, decorative covers."
        )
    )

    # ── Deflection ────────────────────────────────────────────────────────
    deflection_category: DeflectionCategory = Field(
        default=DeflectionCategory.MECHANICAL,
        description=(
            "How tightly deflection must be controlled. "
            "'precision': instruments, optical mounts (L/500). "
            "'structural': civil/architectural frames (L/300). "
            "'mechanical': general machine parts (L/100) — default. "
            "'flexible': covers, housings, aesthetic parts (L/30)."
        )
    )

    # ── Confidence & reasoning ────────────────────────────────────────────
    confidence: float = Field(
        description=(
            "Your confidence in this load estimate, 0.0–1.0. "
            "Set BELOW 0.75 when: "
            "(1) the object description is ambiguous or generic "
            "    (e.g. 'bracket', 'part', 'component' with no context), "
            "(2) the load magnitude could span more than one order of magnitude "
            "    depending on the specific application, "
            "(3) the stress mode is unclear from the description, or "
            "(4) the object could serve very different roles "
            "    (e.g. 'arm' could be a robot arm or a door hinge). "
            "Set HIGH (0.85+) only for well-known, function-specific objects "
            "(bicycle crank, car door hinge, bolt, gear tooth)."
        )
    )

    reasoning: str = Field(
        description=(
            "Concise chain-of-thought explaining the load estimate. "
            "Reference: the object's function, typical operating environment, "
            "known failure modes, and how the geometry supports or challenges "
            "the estimate. 2–4 sentences."
        )
    )

    # ── Clarification ─────────────────────────────────────────────────────
    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "Populate ONLY when confidence < 0.75. "
            "Ask the single question whose answer would most reduce uncertainty. "
            "Make it specific and answerable in one sentence. "
            "Examples: "
            "'Is this bracket structural (load-bearing) or decorative?' "
            "'What is the maximum force this part will experience in operation?' "
            "'Is this a marine, automotive, or indoor environment?' "
            "Do NOT ask multiple questions. "
            "Leave null when confidence >= 0.75."
        )
    )

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {v}. "
                "Use 0.5 for a highly uncertain estimate."
            )
        return round(v, 3)

    @field_validator("safety_factor")
    @classmethod
    def safety_factor_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"safety_factor must be positive, got {v}."
            )
        if v < 1.0:
            raise ValueError(
                f"safety_factor of {v} is below 1.0 — this implies the design "
                "load exceeds expected failure load. Use a minimum of 1.0 "
                "(no safety margin) for the most aggressive aerospace cases."
            )
        return round(v, 2)

    @field_validator("magnitude_range_N")
    @classmethod
    def magnitude_range_valid(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = float(v[0]), float(v[1])
        if lo < 0 or hi < 0:
            raise ValueError(
                f"magnitude_range_N values must be non-negative. Got [{lo}, {hi}]."
            )
        if lo > hi:
            # Swap silently — LLM may return them in wrong order
            lo, hi = hi, lo
        return (lo, hi)

    @field_validator("operating_temp_C")
    @classmethod
    def temp_range_valid(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = float(v[0]), float(v[1])
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    @model_validator(mode="after")
    def clarification_consistency(self) -> "LoadEstimate":
        """
        Ensure clarification_question is populated iff confidence < 0.75,
        and that purely thermal parts use stress_mode='none'.
        """
        if self.confidence < 0.75 and self.clarification_question is None:
            # LLM forgot to generate a question — insert a generic fallback
            self.clarification_question = (
                "Could you describe the primary function of this part and "
                "the maximum force or load it will experience in use?"
            )
        if self.confidence >= 0.75 and self.clarification_question is not None:
            # LLM generated a question despite high confidence — discard it
            self.clarification_question = None

        if (
            self.load_type == LoadType.THERMAL
            and self.primary_stress_mode != StressMode.NONE
            and self.magnitude_range_N == (0.0, 0.0)
        ):
            # Thermal-only part should have stress_mode=none
            self.primary_stress_mode = StressMode.NONE

        return self


# Threshold used by the node to decide whether to route to clarification
CONFIDENCE_THRESHOLD: float = 0.75
