"""
explanation/prompt_builder.py
------------------------------
Prompt construction for the explanation node.

Three static methods:
  build_main_prompt()       -> system prompt  (covers both initial + followup modes)
  build_user_message()      -> initial explanation user message
  build_followup_message()  -> follow-up question user message
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


class BasePromptBuilder(ABC):
    @staticmethod
    @abstractmethod
    def build_main_prompt() -> str:
        pass


class ExplanationPromptBuilder(BasePromptBuilder):

    @staticmethod
    def build_main_prompt() -> str:
        """
        System prompt covering both the initial explanation and follow-up mode.
        The LLM will operate in initial mode or follow-up mode depending on
        the conversation history — same system prompt handles both.
        """
        return """\
You are a senior materials engineer advising a product design engineer \
on eco-material selection. You operate in two modes:

━━━ MODE 1: INITIAL EXPLANATION ━━━

When there is no prior conversation, you have been given the full \
reasoning chain for a material recommendation: geometry, inferred load \
profile, derived mechanical requirements, and ranked ML model predictions.

Write exactly three paragraphs of plain prose. No bullet points, no headers, \
no markdown. Each paragraph 2–4 sentences.

  Paragraph 1 — Geometry and loading:
    How the part's shape and dimensions, combined with the load type and \
stress mode, produced the mechanical requirements. \
Cite the actual geometry values (wall thickness, span, aspect ratio) \
and load values (force range, safety factor).

  Paragraph 2 — Why the top material:
    How the top-ranked material satisfies the required tensile strength and \
stiffness. Cite both the requirement and the material's predicted performance. \
If fatigue, temperature range, or density influenced selection, say so with values.
    — If stiffness_dominated=True: add one sentence noting no material fully \
meets the stiffness need and a geometry change is required (thicker walls, \
shorter span, or ribbing).
    — If buckling_risk=True: add one sentence noting the Euler buckling risk \
and suggesting lateral bracing or a larger cross-section.

  Paragraph 3 — Trade-off vs runner-up:
    Compare the top material to the next-best alternative on eco score and \
the binding mechanical constraint. Name the trade-off the engineer is making.
    — If only one prediction: omit paragraph 3, end paragraph 2 with a note \
that no alternative candidates were returned.

━━━ MODE 2: FOLLOW-UP QUESTIONS ━━━

When there is a prior conversation (you have already provided the initial \
explanation), a design engineer is asking follow-up questions. \
Answer directly and concisely using only information from the pipeline \
context provided. Do not repeat the full explanation.

Rules for follow-up answers:
  — Answer in 2–5 sentences unless the question genuinely requires more.
  — Always cite specific numbers when they are relevant.
  — If the question asks about a scenario the pipeline did not model \
(e.g. 'what if I doubled the wall thickness?'): reason qualitatively \
from the values you do have, and note that a new analysis would be \
needed for a definitive answer.
  — If the question is outside the scope of this pipeline's analysis \
(e.g. supply chain, cost sourcing, manufacturing process): say so clearly \
and redirect to what you can address from the data.
  — Never invent properties or data not present in the context.

━━━ GROUNDING RULES (BOTH MODES) ━━━

  — Every factual claim must trace to a number from the provided context.
  — No marketing language: 'excellent', 'outstanding', 'superior', 'ideal'.
    Use numerical comparisons instead.
  — No hedging: 'appears to', 'may be', 'could potentially'.
    Commit to what the numbers show.\
"""

    @staticmethod
    def build_user_message(
        object_description: str,
        geometry: Dict[str, Any],
        load_estimate: Dict[str, Any],
        ml_input_vector: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        clarification_answer: Optional[str] = None,
    ) -> str:
        """
        User message for the initial explanation call (Mode 1).
        """
        if not predictions:
            raise ValueError(
                "predictions list is empty — cannot build explanation message."
            )

        top       = predictions[0]
        runner_up = predictions[1] if len(predictions) > 1 else None

        lines = ["## Part"]
        lines.append(f"  Description: {object_description.strip()}")
        if clarification_answer:
            lines.append(f"  User clarified: {clarification_answer.strip()}")

        lines += ["", "## Geometry"]
        lines += _fmt_geometry(geometry)

        lines += ["", "## Inferred load profile"]
        lines += _fmt_load(load_estimate)

        lines += ["", "## Derived mechanical requirements"]
        lines += _fmt_requirements(ml_input_vector)

        lines += ["", "## Top recommendation"]
        lines += _fmt_prediction(top, ml_input_vector, rank=1)

        if runner_up:
            lines += ["", "## Runner-up"]
            lines += _fmt_prediction(runner_up, ml_input_vector, rank=2)
        else:
            lines += ["", "## Runner-up", "  (none — only one candidate)"]

        lines += [
            "",
            "Produce the three-paragraph explanation. "
            "Every sentence must cite at least one number from the data above.",
        ]

        return "\n".join(lines)

    @staticmethod
    def build_followup_message(question: str) -> str:
        """
        User message for a follow-up question (Mode 2).
        The pipeline context is already in the conversation history,
        so we only need to append the new question.
        """
        return question.strip()

    @staticmethod
    def build_context_injection(
        object_description: str,
        geometry: Dict[str, Any],
        load_estimate: Dict[str, Any],
        ml_input_vector: Dict[str, Any],
        predictions: List[Dict[str, Any]],
    ) -> str:
        """
        Builds the context block prepended to the FIRST follow-up message
        so the LLM has the pipeline data available without re-reading the
        full initial explanation prompt.

        This is injected as a system-level context message before the
        conversation history when ask_followup() calls the LLM.
        """
        lines = [
            "## Pipeline context for follow-up questions",
            f"  Part: {object_description.strip()}",
            "",
            "## Geometry summary",
        ]
        lines += _fmt_geometry(geometry)

        lines += ["", "## Load profile"]
        lines += _fmt_load(load_estimate)

        lines += ["", "## Mechanical requirements"]
        lines += _fmt_requirements(ml_input_vector)

        lines += ["", "## Predictions"]
        for i, p in enumerate(predictions[:3], 1):
            lines += _fmt_prediction(p, ml_input_vector, rank=i)

        lines += ["", "Use only the values above when answering follow-up questions."]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private formatters (shared between build_user_message and build_context_injection)
# ---------------------------------------------------------------------------

def _fmt_geometry(geo: Dict[str, Any]) -> List[str]:
    lines = []
    x, y, z = geo.get("bbox_x_mm"), geo.get("bbox_y_mm"), geo.get("bbox_z_mm")
    if x and y and z:
        lines.append(f"  Bounding box:   {x:.1f} × {y:.1f} × {z:.1f} mm")
    for label, key, fmt in [
        ("Shape",         "dominant_axis",            "{}"),
        ("Aspect ratio",  "aspect_ratio",             "{:.1f}"),
        ("Min wall",      "min_wall_thickness_mm",    "{:.2f} mm"),
        ("Median wall",   "median_wall_thickness_mm", "{:.2f} mm"),
        ("Thin features", "has_thin_features",        "{}"),
        ("Watertight",    "is_watertight",             "{}"),
        ("Volume",        "volume_mm3",               "{:.0f} mm³"),
    ]:
        v = geo.get(key)
        if v is not None:
            lines.append(f"  {label + ':':16s} {fmt.format(v)}")
    return lines or ["  (no geometry data)"]


def _fmt_load(load: Dict[str, Any]) -> List[str]:
    lines = []
    for label, key in [
        ("Load type",       "load_type"),
        ("Stress mode",     "primary_stress_mode"),
        ("Safety factor",   "safety_factor"),
        ("Fatigue critical","is_fatigue_critical"),
        ("Deflection",      "deflection_category"),
    ]:
        v = load.get(key)
        if v is not None:
            lines.append(f"  {label + ':':18s} {v}")
    mag = load.get("magnitude_range_N")
    if mag:
        lines.append(f"  {'Force range:':18s} {mag[0]:.0f}–{mag[1]:.0f} N")
    temp = load.get("operating_temp_C")
    if temp:
        lines.append(f"  {'Temperature:':18s} {temp[0]:.0f}–{temp[1]:.0f} °C")
    reasoning = load.get("reasoning")
    if reasoning:
        lines.append(f"  {'LLM reasoning:':18s} {reasoning.strip()}")
    return lines or ["  (no load data)"]


def _fmt_requirements(vec: Dict[str, Any]) -> List[str]:
    lines = []
    strength  = vec.get("required_tensile_strength_MPa")
    stiffness = vec.get("required_stiffness_GPa")
    dominated = vec.get("stiffness_dominated", False)
    if strength is not None:
        lines.append(f"  {'Tensile strength:':22s} {strength:.1f} MPa required")
    if stiffness is not None:
        suffix = " [GEOMETRY-DOMINATED]" if dominated else " required"
        lines.append(f"  {'Stiffness:':22s} {stiffness:.1f} GPa{suffix}")
    for label, key in [
        ("Max density",     "max_density_kg_m3"),
        ("Fatigue critical","fatigue_critical"),
        ("Buckling risk",   "buckling_risk"),
    ]:
        v = vec.get(key)
        if v is not None:
            lines.append(f"  {label + ':':22s} {v}")
    t_min = vec.get("min_operating_temp_C")
    t_max = vec.get("max_operating_temp_C")
    if t_min is not None and t_max is not None:
        lines.append(f"  {'Temp range:':22s} {t_min:.0f}–{t_max:.0f} °C")
    deriv = vec.get("derivation", {})
    s = deriv.get("strength", {})
    if s.get("formula"):
        lines.append(f"  {'Strength formula:':22s} {s['formula']}")
        if "F_N" in s and "L_mm" in s:
            lines.append(
                f"  {'  inputs:':22s} F={s['F_N']:.0f} N, "
                f"L={s['L_mm']:.0f} mm, SF={s.get('safety_factor', '?')}"
            )
    e = deriv.get("stiffness", {})
    if e.get("deflection_divisor"):
        lines.append(
            f"  {'Stiffness basis:':22s} L/{e['deflection_divisor']:.0f} "
            f"({e.get('deflection_category', '')})"
        )
    return lines or ["  (no requirements data)"]


def _fmt_prediction(
    pred: Dict[str, Any],
    req: Dict[str, Any],
    rank: int,
) -> List[str]:
    lines = [f"  Material:          {pred.get('material_name', 'Unknown')}"]
    eco  = pred.get("eco_score")
    conf = pred.get("confidence")
    if eco  is not None: lines.append(f"  Eco score:         {eco:.3f}")
    if conf is not None: lines.append(f"  Model confidence:  {conf:.3f}")
    p_str = pred.get("predicted_strength_MPa")
    r_str = req.get("required_tensile_strength_MPa")
    if p_str is not None and r_str is not None:
        meets = ">=" if p_str >= r_str else "< (DOES NOT MEET)"
        lines.append(f"  Pred. strength:    {p_str:.0f} MPa {meets} {r_str:.0f} MPa required")
    elif p_str is not None:
        lines.append(f"  Pred. strength:    {p_str:.0f} MPa")
    p_sti = pred.get("predicted_stiffness_GPa")
    r_sti = req.get("required_stiffness_GPa")
    if p_sti is not None and r_sti is not None:
        meets = ">=" if p_sti >= r_sti else "< (DOES NOT MEET)"
        lines.append(f"  Pred. stiffness:   {p_sti:.1f} GPa {meets} {r_sti:.1f} GPa required")
    elif p_sti is not None:
        lines.append(f"  Pred. stiffness:   {p_sti:.1f} GPa")
    return lines