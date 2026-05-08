"""
translate_features_to_ml_inputs.py
===================================
LangChain tool: translates CAD geometry + load estimate into the
numerical input vector required by the eco-material ML model.

Pipeline
--------
1. Parse & validate inputs          (_parse_geometry, _parse_load_estimate)
2. Detect structural primitives     (archetype_detector.detect_primitives)
3. Distribute loads to primitives   (load_distributor.distribute_loads)
4. Solve mechanics per primitive    (mechanics_solver.solve_primitive)
5. Envelope results                 (max across all primitives)
6. Build output dict

Design goals
------------
- Zero LLM calls. Fully deterministic.
- Each intermediate result is traceable back to a closed-form formula.
- Works for any STL — no object-type assumptions.
- Backwards-compatible output schema with the old tool.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

from langchain.tools import tool

try:
    from .structural_primitives import ArchetypeKind, StructuralPrimitive
    from .archetype_detector import detect_primitives
    from .load_distributor import distribute_loads
    from .mechanics_solver import solve_primitive
except ImportError:  # pragma: no cover - supports direct script execution in tools/files
    from structural_primitives import ArchetypeKind, StructuralPrimitive
    from archetype_detector import detect_primitives
    from load_distributor import distribute_loads
    from mechanics_solver import solve_primitive

logger = logging.getLogger("feature-translation")

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_REALISTIC_STIFFNESS_GPA = 500.0

_DEFLECTION_DIVISORS: dict[str, float] = {
    "precision":   500.0,
    "structural":  300.0,
    "mechanical":  150.0,
    "flexible":     30.0,
}
_DEFAULT_DEFLECTION_CATEGORY = "mechanical"

_VALID_LOAD_TYPES   = {"static", "cyclic", "impact", "thermal"}
_VALID_STRESS_MODES = {"bending", "tension", "compression", "torsion", "none"}


# ── Tool ──────────────────────────────────────────────────────────────────────

@tool
def translate_features_to_ml_inputs(
    geometry: dict,
    load_estimate: dict,
    recyclability_priority: float = 0.7,
) -> dict:
    """
    Translate CAD geometry features and inferred load estimate into the
    numerical input vector expected by the eco-material ML model.

    Parameters
    ----------
    geometry :
        Output of parse_and_extract_cad_features (trimesh analysis).
        Required keys:
            bbox_x_mm, bbox_y_mm, bbox_z_mm, dominant_axis
        Optional (improve accuracy when present):
            min_wall_thickness_mm, median_wall_thickness_mm,
            is_watertight, connected_components, mean_curvature,
            bottom_contact_area_mm2, top_load_area_mm2, volume_mm3,
            has_cylindrical_bores

    load_estimate :
        Output of the load_inference node.
        Required keys:
            load_type, primary_stress_mode, magnitude_range_N,
            safety_factor, operating_temp_C, is_fatigue_critical
        Optional:
            deflection_category

    recyclability_priority :
        Eco weighting 0.0 (performance) … 1.0 (max recyclability).

    Returns
    -------
    dict with keys:
        required_tensile_strength_MPa   float
        required_stiffness_GPa          float
        stiffness_dominated             bool
        buckling_risk                   bool
        min_operating_temp_C            float
        max_operating_temp_C            float
        max_density_kg_m3               float
        fatigue_critical                bool
        recyclability_priority          float
        governing_primitive             str   archetype of worst-case primitive
        primitives                      list  per-primitive analysis summaries
        derivation                      dict
        warnings                        list[str]
    """
    logger.info("[Tool] translate_features_to_ml_inputs — START")
    warnings: list[str] = []

    # ── 1. Parse inputs ───────────────────────────────────────────────────
    if not 0.0 <= recyclability_priority <= 1.0:
        raise ValueError(
            f"recyclability_priority must be in [0.0, 1.0], "
            f"got {recyclability_priority}."
        )

    load = _parse_load_estimate(load_estimate, warnings)

    # ── 2. Detect structural primitives ───────────────────────────────────
    primitives = detect_primitives(geometry, load, warnings)
    logger.info("Detected %d primitive(s): %s",
                len(primitives), [p.kind.value for p in primitives])

    # ── 3. Distribute loads ───────────────────────────────────────────────
    distribute_loads(primitives, load, geometry, warnings)

    # ── 4. Solve mechanics per primitive ──────────────────────────────────
    deflection_cat = load["deflection_category"]
    SF             = load["safety_factor"]

    for prim in primitives:
        solve_primitive(prim, load, deflection_cat, SF, warnings)
        warnings.extend(prim.warnings)

    # ── 5. Thermal-only short-circuit ─────────────────────────────────────
    is_thermal_only = (
        load["load_type"] == "thermal"
        and load["primary_stress_mode"] == "none"
    )
    if is_thermal_only:
        warnings.append(
            "Thermal-only load: mechanical strength/stiffness requirements "
            "are zero. Material selection will be driven by operating "
            "temperature range and density budget."
        )
        return _thermal_only_output(load, recyclability_priority, warnings)

    # ── 6. Envelope — pick governing primitive ────────────────────────────
    governing = _envelope(primitives, warnings)

    # Stiffness dominated?
    stiffness_dominated = governing.required_E_GPa >= _MAX_REALISTIC_STIFFNESS_GPA
    if stiffness_dominated:
        warnings.append(
            f"Required stiffness ({governing.required_E_GPa:.0f} GPa) at or "
            f"above the material cap ({_MAX_REALISTIC_STIFFNESS_GPA:.0f} GPa). "
            "Geometry — not material — is the limiting factor. "
            "Consider increasing wall thickness, shortening span, or adding ribs."
        )

    # ── 7. Density budget ─────────────────────────────────────────────────
    max_density = _density_budget(load)

    # ── 8. Assemble output ────────────────────────────────────────────────
    result = {
        "required_tensile_strength_MPa": round(governing.required_strength_MPa, 2),
        "required_stiffness_GPa":        round(
            min(governing.required_E_GPa, _MAX_REALISTIC_STIFFNESS_GPA), 3
        ),
        "stiffness_dominated":           stiffness_dominated,
        "buckling_risk":                 any(p.buckling_risk for p in primitives),
        "min_operating_temp_C":          load["operating_temp_C"][0],
        "max_operating_temp_C":          load["operating_temp_C"][1],
        "max_density_kg_m3":             round(max_density, 1),
        "fatigue_critical":              load["is_fatigue_critical"],
        "recyclability_priority":        recyclability_priority,
        "governing_primitive":           governing.kind.value,
        "primitives":                    [p.summary() for p in primitives],
        "derivation": {
            "governing_primitive_kind":  governing.kind.value,
            "governing_formula":         governing.governing_formula,
            "governing_derivation":      governing.derivation,
            "all_primitives_summary": [
                {
                    "archetype":           p.kind.value,
                    "required_strength":   round(p.required_strength_MPa, 2),
                    "required_E_GPa":      round(p.required_E_GPa, 3),
                    "buckling_risk":       p.buckling_risk,
                }
                for p in primitives
            ],
        },
        "warnings": warnings,
    }

    logger.info(
        "[Tool] DONE — strength=%.1f MPa, stiffness=%.2f GPa, "
        "governing=%s, buckling=%s",
        result["required_tensile_strength_MPa"],
        result["required_stiffness_GPa"],
        result["governing_primitive"],
        result["buckling_risk"],
    )
    return result


# ── Envelope ───────────────────────────────────────────────────────────────────

def _envelope(
    primitives: list[StructuralPrimitive],
    warnings: list[str],
) -> StructuralPrimitive:
    """
    Select the governing primitive — the one with the highest strength
    requirement (most demanding from a material-selection perspective).

    Tie-break: highest stiffness requirement.
    """
    if not primitives:
        raise RuntimeError("No primitives to envelope — this is a bug.")

    governing = max(
        primitives,
        key=lambda p: (p.required_strength_MPa, p.required_E_GPa),
    )

    if len(primitives) > 1:
        logger.debug(
            "Governing primitive: %s (%.1f MPa, %.2f GPa)",
            governing.kind.value,
            governing.required_strength_MPa,
            governing.required_E_GPa,
        )

    return governing


# ── Input parsing ──────────────────────────────────────────────────────────────

def _parse_load_estimate(raw: dict, warnings: list[str]) -> dict:
    required = [
        "load_type", "primary_stress_mode", "magnitude_range_N",
        "safety_factor", "operating_temp_C", "is_fatigue_critical",
    ]
    for key in required:
        if key not in raw:
            raise ValueError(
                f"load_estimate missing required key '{key}'."
            )

    load_type = raw["load_type"]
    if load_type not in _VALID_LOAD_TYPES:
        raise ValueError(f"load_type must be one of {_VALID_LOAD_TYPES}. Got '{load_type}'.")

    stress_mode = raw["primary_stress_mode"]
    if stress_mode not in _VALID_STRESS_MODES:
        raise ValueError(
            f"primary_stress_mode must be one of {_VALID_STRESS_MODES}. "
            f"Got '{stress_mode}'."
        )

    mag = raw["magnitude_range_N"]
    if not (isinstance(mag, (list, tuple)) and len(mag) == 2):
        raise TypeError(f"magnitude_range_N must be [min, max]. Got {mag!r}.")
    mag_min, mag_max = float(mag[0]), float(mag[1])
    if mag_min < 0 or mag_max < 0:
        raise ValueError(f"magnitude_range_N must be non-negative. Got [{mag_min}, {mag_max}].")
    if mag_min > mag_max:
        warnings.append("magnitude_range_N: min > max, swapping.")
        mag_min, mag_max = mag_max, mag_min

    sf = float(raw["safety_factor"])
    if sf <= 0:
        raise ValueError(f"safety_factor must be positive. Got {sf}.")
    if sf < 1.0:
        warnings.append(
            f"safety_factor={sf} < 1.0 — design load exceeds expected "
            "failure load. Likely an inference error."
        )

    temp = raw["operating_temp_C"]
    if not (isinstance(temp, (list, tuple)) and len(temp) == 2):
        raise TypeError(f"operating_temp_C must be [min, max]. Got {temp!r}.")
    temp_min, temp_max = float(temp[0]), float(temp[1])
    if temp_min > temp_max:
        warnings.append("operating_temp_C: min > max, swapping.")
        temp_min, temp_max = temp_max, temp_min

    deflection_category = raw.get("deflection_category", _DEFAULT_DEFLECTION_CATEGORY)
    if deflection_category not in _DEFLECTION_DIVISORS:
        warnings.append(
            f"Unknown deflection_category '{deflection_category}'. "
            f"Using '{_DEFAULT_DEFLECTION_CATEGORY}'."
        )
        deflection_category = _DEFAULT_DEFLECTION_CATEGORY

    if load_type == "thermal" and stress_mode not in {"none", "compression"}:
        warnings.append(
            f"load_type='thermal' but stress_mode='{stress_mode}'. "
            "For purely thermal loads use stress_mode='none'."
        )

    return {
        "load_type":            load_type,
        "primary_stress_mode":  stress_mode,
        "mag_max_N":            mag_max,
        "mag_min_N":            mag_min,
        "safety_factor":        sf,
        "operating_temp_C":     (temp_min, temp_max),
        "is_fatigue_critical":  bool(raw["is_fatigue_critical"]),
        "deflection_category":  deflection_category,
    }


# ── Density budget ─────────────────────────────────────────────────────────────

def _density_budget(load: dict) -> float:
    """
    Order-of-magnitude maximum density guideline for the ML model.
    These are soft constraints used to weight the eco objective.
    """
    lt = load["load_type"]
    if lt == "thermal":
        return 10_000.0
    if lt == "static":
        return 8_000.0
    if lt in ("cyclic", "impact"):
        return 4_500.0
    return 7_000.0


# ── Thermal-only output ────────────────────────────────────────────────────────

def _thermal_only_output(
    load: dict,
    recyclability_priority: float,
    warnings: list[str],
) -> dict:
    return {
        "required_tensile_strength_MPa": 0.0,
        "required_stiffness_GPa":        0.0,
        "stiffness_dominated":           False,
        "buckling_risk":                 False,
        "min_operating_temp_C":          load["operating_temp_C"][0],
        "max_operating_temp_C":          load["operating_temp_C"][1],
        "max_density_kg_m3":             10_000.0,
        "fatigue_critical":              load["is_fatigue_critical"],
        "recyclability_priority":        recyclability_priority,
        "governing_primitive":           "none",
        "primitives":                    [],
        "derivation":                    {"note": "Thermal load only. No mechanical analysis."},
        "warnings":                      warnings,
    }
