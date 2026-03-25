from __future__ import annotations

from typing import Any, Dict, Optional

from abc import ABC, abstractmethod
from agents.base_prompt_builder import BasePromptBuilder

class LoadInferencePromptBuilder(BasePromptBuilder):
    """
    Builds prompts for the load_inference node.
    All methods are static — no instance state is needed.
    """

    @staticmethod
    def build_main_prompt() -> str:
        """
        System prompt: role, task, output format rules, confidence calibration.
        This is passed to BaseAgentNode.__init__ as system_prompt.
        """
        return """You are a senior mechanical engineer specialising in structural \
analysis and materials selection. Your task is to infer the mechanical load \
profile for an engineering part given:
  1. A plain-language description of what the object is
  2. Key geometric features extracted from its CAD file
  3. Optionally, a clarifying answer the user has provided

You will produce a structured LoadEstimate that will be used downstream to \
derive required material properties (strength, stiffness, temperature range) \
for an eco-material selection model.

━━━ OUTPUT REQUIREMENTS ━━━

Your output must be a valid LoadEstimate with the following fields:

load_type           — "static" | "cyclic" | "impact" | "thermal"
primary_stress_mode — "bending" | "tension" | "compression" | "torsion" | "none"
magnitude_range_N   — [min_force_N, max_force_N] typical operating range
safety_factor       — appropriate for this object class and industry
operating_temp_C    — [min_C, max_C] expected environment
is_fatigue_critical — true if >10,000 cycles AND safety-critical
deflection_category — "precision" | "structural" | "mechanical" | "flexible"
confidence          — 0.0–1.0 (see calibration rules below)
reasoning           — 2–4 sentences explaining your inference
clarification_question — null if confidence ≥ 0.75, one question otherwise

━━━ REASONING APPROACH ━━━

Follow this reasoning chain:
  1. Identify the object class from the description (e.g. "bicycle crank arm",
     "structural bracket", "gear shaft")
  2. Recall the canonical function of that object class — what forces act on it,
     in what direction, with what frequency
  3. Cross-check against the geometry: does the shape (elongated/flat/compact,
     wall thickness, aspect ratio) match the expected load-bearing form?
     Flag mismatches in your reasoning.
  4. Assign a load magnitude range from your engineering knowledge of that
     object class. Use typical operating loads, not ultimate failure loads.
  5. Assign a safety factor appropriate to the industry/application context
     implied by the description.

━━━ CONFIDENCE CALIBRATION ━━━

Set confidence BELOW 0.75 (and populate clarification_question) when:
  — The description is generic: "bracket", "part", "arm", "housing" with no
    industry or function context
  — The load could span >1 order of magnitude depending on application
    (e.g. "hook" could be 10 N or 100,000 N)
  — The geometry contradicts the expected shape for the stated object class
    (e.g. "beam" described as compact with aspect_ratio ≈ 1.0)
  — The stress mode is ambiguous from the description alone

Set confidence 0.75–0.85 when you are reasonably certain of the object class
but have some uncertainty about the specific application or scale.

Set confidence above 0.85 only for well-specified, function-clear objects:
"bicycle crank arm", "M8 bolt in shear", "automotive door hinge",
"drone motor mount".

━━━ CLARIFICATION QUESTIONS ━━━

When asking a clarification question:
  — Ask the SINGLE question that most reduces uncertainty
  — Make it answerable in one sentence
  — Do not ask for numbers unless the user is an engineer
  — Prefer functional questions: "Is this load-bearing or decorative?"
    over technical ones: "What is the Von Mises stress?"

━━━ UNITS & CONVENTIONS ━━━

  — Forces in Newtons (N)
  — Temperatures in Celsius (°C)
  — Geometry inputs are in millimetres (mm)
  — Do NOT convert units — output exactly as specified above"""

    @staticmethod
    def build_user_message(
        object_description: str,
        geometry: Dict[str, Any],
        clarification_answer: Optional[str] = None,
    ) -> str:
        """
        Build the per-call user message.

        Args:
            object_description:
                The user's plain-language description of the part.
                e.g. "bicycle crank arm", "structural bracket for a shelf unit"
            geometry:
                The GeometryFeatures dict from parse_and_extract_cad_features.
                We extract the most structurally relevant fields here —
                passing the full dict would include noise (triangle count,
                centroid, etc.) that doesn't help load inference.
            clarification_answer:
                The user's reply to a previous clarification question, if any.
                None on the first invocation.

        Returns:
            A formatted string ready to use as the 'user' role message.
        """
        # Extract the geometry fields that matter for load inference
        # Deliberately omit centroid, triangle_count, vertex_count — noise
        geo_summary = _format_geometry(geometry)

        lines = ["## Part description", object_description.strip(), ""]

        lines += ["## Geometry (from CAD)", geo_summary, ""]

        if clarification_answer:
            lines += [
                "## User clarification",
                f"The user answered: {clarification_answer.strip()}",
                "",
                "Incorporate this answer to refine your load estimate. "
                "Your confidence should now be ≥ 0.75 unless the answer "
                "itself introduces new ambiguity.",
                "",
            ]

        lines.append(
            "Based on the part description and geometry above, produce a "
            "LoadEstimate. Reason from first principles about the object's "
            "function before assigning load values."
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Geometry formatting helper
# ---------------------------------------------------------------------------

def _format_geometry(geo: Dict[str, Any]) -> str:
    """
    Format the geometry dict into a concise, readable summary.
    Only includes fields relevant to load inference.
    """
    lines = []

    # Bounding box
    x = geo.get("bbox_x_mm")
    y = geo.get("bbox_y_mm")
    z = geo.get("bbox_z_mm")
    if x is not None and y is not None and z is not None:
        lines.append(f"  Bounding box:     {x:.1f} × {y:.1f} × {z:.1f} mm")

    # Dominant shape
    axis = geo.get("dominant_axis")
    ar   = geo.get("aspect_ratio")
    if axis:
        ar_str = f"  (aspect ratio {ar:.1f})" if ar is not None else ""
        lines.append(f"  Shape class:      {axis}{ar_str}")

    # Wall thickness
    t_min = geo.get("min_wall_thickness_mm")
    t_med = geo.get("median_wall_thickness_mm")
    if t_min is not None:
        lines.append(f"  Min wall:         {t_min:.2f} mm")
    if t_med is not None:
        lines.append(f"  Median wall:      {t_med:.2f} mm")

    # Thin features flag
    thin = geo.get("has_thin_features")
    if thin is not None:
        lines.append(f"  Thin features:    {'yes' if thin else 'no'}")

    # Watertight
    wt = geo.get("is_watertight")
    if wt is not None:
        lines.append(
            f"  Mesh watertight:  {'yes' if wt else 'no (open shell — wall thickness estimates less reliable)'}"
        )

    # Volume
    vol = geo.get("volume_mm3")
    if vol is not None:
        lines.append(f"  Volume:           {vol:.1f} mm³")

    # Any warnings from the CAD parser (geometry anomalies the LLM should know)
    parser_warnings = geo.get("warnings", [])
    structural_warnings = [
        w for w in parser_warnings
        if any(kw in w.lower() for kw in
               ["watertight", "winding", "degenerate", "thin", "open"])
    ]
    if structural_warnings:
        lines.append("  CAD parser notes:")
        for w in structural_warnings:
            lines.append(f"    - {w}")

    return "\n".join(lines) if lines else "  (geometry not available)"
