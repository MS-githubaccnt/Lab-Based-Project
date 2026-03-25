from __future__ import annotations

import math
from typing import Optional
from langchain.tools import tool
import logging

logger = logging.getLogger("roster-neural")

_MAX_REALISTIC_STIFFNESS_GPA = 500.0

# Deflection limit as a fraction of span (L / divisor)
_DEFLECTION_DIVISORS: dict[str, float] = {
    "precision":  500.0,
    "structural": 300.0,
    "mechanical": 100.0,
    "flexible":    30.0,
}
_DEFAULT_DEFLECTION_CATEGORY = "mechanical"

# Minimum cross-section area to avoid division-by-zero on degenerate meshes
_MIN_AREA_MM2 = 0.01


@tool
def translate_features_to_ml_inputs(
    geometry: dict,
    load_estimate: dict,
    recyclability_priority: float = 0.7,
) -> dict:
    """
    Translate CAD geometry features and inferred load estimate into
    the numerical input vector expected by the eco-material ML model.

    Args:
        geometry:
            Output dict of parse_and_extract_cad_features.
            Required keys: bbox_x_mm, bbox_y_mm, bbox_z_mm,
            min_wall_thickness_mm, median_wall_thickness_mm,
            dominant_axis, is_watertight.

        load_estimate:
            Output dict of the load_inference LLM node.
            Required keys: load_type, primary_stress_mode,
            magnitude_range_N, safety_factor, operating_temp_C,
            is_fatigue_critical.
            Optional keys: deflection_category.

        recyclability_priority:
            User-supplied eco weighting from 0.0 (performance only)
            to 1.0 (maximum recyclability). Default 0.7.

    Returns dict with keys:

        # Mechanical requirements
        required_tensile_strength_MPa   float
        required_stiffness_GPa          float
        stiffness_dominated             bool   True = geometry is the
                                               real constraint, not material
        buckling_risk                   bool   True = Euler buckling check
                                               flagged for compression loads

        # Thermal
        min_operating_temp_C            float
        max_operating_temp_C            float

        # Other
        max_density_kg_m3               float
        fatigue_critical                bool
        recyclability_priority          float

        # Traceability — how each number was derived
        derivation                      dict   (field -> formula + values used)

        # Non-fatal issues
        warnings                        list[str]

    Raises:
        ValueError  missing or invalid field in geometry or load_estimate
        TypeError   wrong types for required fields
    """
    logger.info("[Tool] Executing translate_features_to_ml_inputs")
    warnings: list[str] = []

    geo  = _parse_geometry(geometry, warnings)
    load = _parse_load_estimate(load_estimate, warnings)

    # Resolve wall thickness — prefer measured values, fall back to bbox estimate
    t_min, t_med = _resolve_wall_thickness(geo, warnings)

    # Span = dominant bounding-box dimension
    span_mm = _dominant_span(geo)

    # ── Stress ────────────────────────────────────────────────────────────
    strength_MPa, strength_derivation = _compute_required_strength(
        load=load,
        t_min_mm=t_min,
        t_med_mm=t_med,
        span_mm=span_mm,
        bbox=geo["bbox"],
        dominant_axis=geo["dominant_axis"],
        warnings=warnings,
    )

    # ── Stiffness ─────────────────────────────────────────────────────────
    stiffness_GPa, stiffness_dominated, stiffness_derivation = _compute_required_stiffness(
        load=load,
        t_med_mm=t_med,
        span_mm=span_mm,
        bbox=geo["bbox"],
        dominant_axis=geo["dominant_axis"],
        warnings=warnings,
    )

    # ── Buckling (compression only) ───────────────────────────────────────
    buckling_risk = False
    if load["primary_stress_mode"] == "compression":
        buckling_risk, buckling_warning = _check_euler_buckling(
            load=load,
            t_med_mm=t_med,
            span_mm=span_mm,
            bbox=geo["bbox"],
        )
        if buckling_warning:
            warnings.append(buckling_warning)

    # ── Density budget ────────────────────────────────────────────────────
    max_density = _estimate_density_budget(load, geo)

    # ── Validate recyclability_priority ──────────────────────────────────
    if not 0.0 <= recyclability_priority <= 1.0:
        raise ValueError(
            f"recyclability_priority must be between 0.0 and 1.0, "
            f"got {recyclability_priority}."
        )

    return {
        "required_tensile_strength_MPa": round(strength_MPa, 2),
        "required_stiffness_GPa":        round(stiffness_GPa, 3),
        "stiffness_dominated":           stiffness_dominated,
        "buckling_risk":                 buckling_risk,
        "min_operating_temp_C":          load["operating_temp_C"][0],
        "max_operating_temp_C":          load["operating_temp_C"][1],
        "max_density_kg_m3":             round(max_density, 1),
        "fatigue_critical":              load["is_fatigue_critical"],
        "recyclability_priority":        recyclability_priority,
        "derivation": {
            "strength": strength_derivation,
            "stiffness": stiffness_derivation,
        },
        "warnings": warnings,
    }


# ===========================================================================
# Input parsing & validation
# ===========================================================================

def _parse_geometry(raw: dict, warnings: list[str]) -> dict:
    """Validate and normalise the geometry dict."""
    required = [
        "bbox_x_mm", "bbox_y_mm", "bbox_z_mm", "dominant_axis",
    ]
    for key in required:
        if key not in raw:
            raise ValueError(
                f"geometry dict is missing required key '{key}'. "
                "Pass the output of parse_and_extract_cad_features directly."
            )

    bbox = (
        float(raw["bbox_x_mm"]),
        float(raw["bbox_y_mm"]),
        float(raw["bbox_z_mm"]),
    )

    if any(d <= 0 for d in bbox):
        raise ValueError(
            f"All bounding-box dimensions must be positive. Got {bbox}. "
            "The CAD file may contain degenerate geometry."
        )

    dominant_axis = raw["dominant_axis"]
    if dominant_axis not in {"elongated", "flat", "compact", "degenerate"}:
        raise ValueError(
            f"dominant_axis must be one of 'elongated', 'flat', 'compact', "
            f"'degenerate'. Got '{dominant_axis}'."
        )

    if dominant_axis == "degenerate":
        warnings.append(
            "Geometry is classified as 'degenerate' (near-zero bounding-box "
            "dimension). Structural calculations will use fallback assumptions "
            "and results will be less reliable."
        )

    # Wall thickness — may be None for open meshes
    t_min = raw.get("min_wall_thickness_mm")
    t_med = raw.get("median_wall_thickness_mm")

    if t_min is not None:
        t_min = float(t_min)
        if t_min <= 0:
            warnings.append(
                f"min_wall_thickness_mm is {t_min} mm (non-positive). "
                "Treating as None and using bbox fallback."
            )
            t_min = None

    if t_med is not None:
        t_med = float(t_med)
        if t_med <= 0:
            t_med = None

    if not raw.get("is_watertight", True):
        warnings.append(
            "Mesh is not watertight — wall thickness measurements may be "
            "unreliable. Structural calculations are approximate."
        )

    return {
        "bbox": bbox,
        "dominant_axis": dominant_axis,
        "t_min": t_min,
        "t_med": t_med,
    }


def _parse_load_estimate(raw: dict, warnings: list[str]) -> dict:
    """Validate and normalise the load_estimate dict."""
    required = [
        "load_type", "primary_stress_mode", "magnitude_range_N",
        "safety_factor", "operating_temp_C", "is_fatigue_critical",
    ]
    for key in required:
        if key not in raw:
            raise ValueError(
                f"load_estimate dict is missing required key '{key}'. "
                "Pass the output of the load_inference node directly."
            )

    load_type = raw["load_type"]
    valid_load_types = {"static", "cyclic", "impact", "thermal"}
    if load_type not in valid_load_types:
        raise ValueError(
            f"load_type must be one of {valid_load_types}. Got '{load_type}'."
        )

    stress_mode = raw["primary_stress_mode"]
    valid_modes = {"bending", "tension", "compression", "torsion", "none"}
    if stress_mode not in valid_modes:
        raise ValueError(
            f"primary_stress_mode must be one of {valid_modes}. "
            f"Got '{stress_mode}'."
        )

    mag = raw["magnitude_range_N"]
    if not (isinstance(mag, (list, tuple)) and len(mag) == 2):
        raise TypeError(
            "magnitude_range_N must be a list or tuple of two floats "
            f"[min_N, max_N]. Got {mag!r}."
        )
    mag_min, mag_max = float(mag[0]), float(mag[1])
    if mag_min < 0 or mag_max < 0:
        raise ValueError(
            f"magnitude_range_N values must be non-negative. Got [{mag_min}, {mag_max}]."
        )
    if mag_min > mag_max:
        warnings.append(
            f"magnitude_range_N min ({mag_min} N) > max ({mag_max} N). "
            "Values have been swapped."
        )
        mag_min, mag_max = mag_max, mag_min

    sf = float(raw["safety_factor"])
    if sf <= 0:
        raise ValueError(
            f"safety_factor must be positive. Got {sf}. "
            "Typical values: 1.5 (aerospace) to 4.0 (consumer products)."
        )
    if sf < 1.0:
        warnings.append(
            f"safety_factor of {sf} is below 1.0 — this means the design "
            "load already exceeds the expected failure load. "
            "This is almost certainly an error in the load inference."
        )

    temp = raw["operating_temp_C"]
    if not (isinstance(temp, (list, tuple)) and len(temp) == 2):
        raise TypeError(
            "operating_temp_C must be a list [min_C, max_C]. "
            f"Got {temp!r}."
        )
    temp_min, temp_max = float(temp[0]), float(temp[1])
    if temp_min > temp_max:
        warnings.append(
            f"operating_temp_C min ({temp_min}°C) > max ({temp_max}°C). "
            "Values have been swapped."
        )
        temp_min, temp_max = temp_max, temp_min

    deflection_category = raw.get("deflection_category", _DEFAULT_DEFLECTION_CATEGORY)
    if deflection_category not in _DEFLECTION_DIVISORS:
        warnings.append(
            f"Unknown deflection_category '{deflection_category}'. "
            f"Using default '{_DEFAULT_DEFLECTION_CATEGORY}'. "
            f"Valid categories: {list(_DEFLECTION_DIVISORS)}."
        )
        deflection_category = _DEFAULT_DEFLECTION_CATEGORY

    if load_type == "thermal" and stress_mode not in {"none", "compression"}:
        warnings.append(
            "load_type is 'thermal' but primary_stress_mode is "
            f"'{stress_mode}'. For purely thermal loads use stress_mode='none'. "
            "Proceeding with provided stress mode."
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


# ===========================================================================
# Wall thickness resolution
# ===========================================================================

def _resolve_wall_thickness(
    geo: dict,
    warnings: list[str],
) -> tuple[float, float]:
    """
    Return (t_min_mm, t_med_mm) for structural calculations.

    Priority:
      1. Measured values from ray-casting (most accurate)
      2. Fallback: bbox_z_mm / 10  (conservative proxy for thin dimension)
         This is used when the mesh is open or ray-casting found no hits.
    """
    t_min = geo["t_min"]
    t_med = geo["t_med"]
    bbox  = geo["bbox"]

    if t_min is None or t_med is None:
        # Use the smallest bounding-box dimension / 10 as a proxy
        thin_dim = min(bbox)
        fallback = thin_dim / 10.0
        fallback = max(fallback, 0.1)  # never below 0.1 mm

        if t_min is None:
            warnings.append(
                f"min_wall_thickness_mm not available (open mesh or failed "
                f"ray-casting). Using bbox fallback: {fallback:.2f} mm "
                f"(smallest bbox dim {thin_dim:.1f} mm / 10). "
                "Strength calculation will be less accurate."
            )
            t_min = fallback

        if t_med is None:
            # Use thin_dim / 4 for median (less conservative than /10)
            med_fallback = max(thin_dim / 4.0, 0.1)
            warnings.append(
                f"median_wall_thickness_mm not available. "
                f"Using bbox fallback: {med_fallback:.2f} mm. "
                "Stiffness calculation will be less accurate."
            )
            t_med = med_fallback

    # Sanity: median should be >= min
    if t_med < t_min:
        warnings.append(
            f"median_wall ({t_med:.2f} mm) < min_wall ({t_min:.2f} mm). "
            "Using min_wall for both to avoid inconsistency."
        )
        t_med = t_min

    return t_min, t_med


# ===========================================================================
# Dominant span
# ===========================================================================

def _dominant_span(geo: dict) -> float:
    """
    Return the structurally relevant span for beam/deflection calculations.

    For elongated parts: largest bbox dimension (the beam length).
    For flat parts:      largest bbox dimension (plate span).
    For compact parts:   largest bbox dimension (conservative).
    """
    return max(geo["bbox"])


# ===========================================================================
# Strength calculation
# ===========================================================================

def _compute_required_strength(
    load: dict,
    t_min_mm: float,
    t_med_mm: float,
    span_mm: float,
    bbox: tuple,
    dominant_axis: str,
    warnings: list[str],
) -> tuple[float, dict]:
    """
    Compute required tensile strength in MPa.

    Uses t_min_mm (worst-case section) throughout.
    Applies safety factor to yield the design requirement.

    Returns (strength_MPa, derivation_dict).
    """
    F   = load["mag_max_N"]
    SF  = load["safety_factor"]
    mode = load["primary_stress_mode"]

    # Pure thermal load — mechanical strength not the driver
    if mode == "none" or (load["load_type"] == "thermal" and F == 0.0):
        return 0.0, {
            "formula": "thermal_only",
            "note": "No mechanical load. Strength requirement driven by temp range only.",
        }

    # Zero load edge case
    if F == 0.0:
        warnings.append(
            "magnitude_range_N max is 0 N. "
            "Returning strength = 0 MPa. If this is a decorative / "
            "non-structural part, this is correct."
        )
        return 0.0, {"formula": "zero_load", "F_N": 0.0}

    dims    = sorted(bbox)          # [min, mid, max]
    b_mm    = dims[1]               # width for bending formula

    if mode == "bending":
        # Cantilever bending: sigma = 6 * F * L / (b * t^2)
        # Conservative: full span as cantilever length.
        # t = min_wall (critical section), b = mid bbox dimension.
        sigma = _sigma_bending(F, span_mm, b_mm, t_min_mm)
        derivation = {
            "formula": "cantilever_bending  sigma = 6·F·L / (b·t²)",
            "F_N": F,
            "L_mm": span_mm,
            "b_mm": b_mm,
            "t_mm": t_min_mm,
            "sigma_MPa": round(sigma, 2),
            "safety_factor": SF,
        }

    elif mode in ("tension", "compression"):
        # sigma = F / A   where A = min_wall * mid_dim (conservative section)
        A_mm2 = max(t_min_mm * b_mm, _MIN_AREA_MM2)
        sigma = F / A_mm2
        derivation = {
            "formula": "axial  sigma = F / A",
            "F_N": F,
            "A_mm2": round(A_mm2, 4),
            "t_mm": t_min_mm,
            "b_mm": b_mm,
            "sigma_MPa": round(sigma, 2),
            "safety_factor": SF,
        }

    elif mode == "torsion":
        # Approximate shaft as solid circle with radius = t_min / 2
        # tau = T * r / J   J = pi * r^4 / 2
        # Equivalent tensile stress via von Mises: sigma_eq = tau * sqrt(3)
        r = t_min_mm / 2.0
        if r < 1e-6:
            r = 1e-6
        # Torque: treat F as tangential force at radius = span/2
        # (conservative: assume moment arm = half span)
        T_Nmm = F * (span_mm / 2.0)
        J     = math.pi * r**4 / 2.0
        tau   = T_Nmm * r / J
        sigma = tau * math.sqrt(3.0)   # von Mises
        derivation = {
            "formula": "torsion  tau=T·r/J, sigma_eq=tau·√3 (von Mises)",
            "T_Nmm": round(T_Nmm, 2),
            "r_mm": round(r, 4),
            "J_mm4": round(J, 4),
            "tau_MPa": round(tau, 2),
            "sigma_eq_MPa": round(sigma, 2),
            "safety_factor": SF,
        }

    else:
        # Fallback: bending is the most conservative assumption
        warnings.append(
            f"Unrecognised primary_stress_mode '{mode}'. "
            "Falling back to bending formula (conservative)."
        )
        sigma = _sigma_bending(F, span_mm, b_mm, t_min_mm)
        derivation = {
            "formula": "fallback_bending",
            "sigma_MPa": round(sigma, 2),
            "safety_factor": SF,
        }

    # Apply safety factor
    required = sigma * SF

    derivation["required_tensile_strength_MPa"] = round(required, 2)
    return required, derivation


def _sigma_bending(F: float, L: float, b: float, t: float) -> float:
    """sigma = 6·F·L / (b·t²) — cantilever bending at root."""
    denom = b * t**2
    if denom < 1e-12:
        return 0.0
    return (6.0 * F * L) / denom


# ===========================================================================
# Stiffness calculation
# ===========================================================================

def _compute_required_stiffness(
    load: dict,
    t_med_mm: float,
    span_mm: float,
    bbox: tuple,
    dominant_axis: str,
    warnings: list[str],
) -> tuple[float, bool, dict]:
    """
    Compute required Young's modulus in GPa.

    Uses t_med_mm (bulk cross-section) — stiffness is governed by the
    full section, not just the thinnest point.

    Returns (E_GPa, stiffness_dominated_bool, derivation_dict).
    """
    F    = load["mag_max_N"]
    mode = load["primary_stress_mode"]
    div  = _DEFLECTION_DIVISORS[load["deflection_category"]]

    if mode == "none" or (load["load_type"] == "thermal" and F == 0.0):
        return 0.0, False, {
            "formula": "thermal_only",
            "note": "No mechanical load.",
        }

    if F == 0.0:
        return 0.0, False, {"formula": "zero_load"}

    dims = sorted(bbox)     # [min, mid, max]
    b_mm = dims[1]          # width

    # Second moment of area — rectangular section, conservative
    I_mm4 = (b_mm * t_med_mm**3) / 12.0
    if I_mm4 < 1e-12:
        warnings.append(
            "Computed second moment of area I is near-zero. "
            "Stiffness calculation unreliable."
        )
        return 0.0, False, {"formula": "degenerate_I"}

    delta_limit_mm = span_mm / div

    # Cantilever: delta = F·L³ / (3·E·I)  =>  E = F·L³ / (3·delta·I)
    E_MPa = (F * span_mm**3) / (3.0 * delta_limit_mm * I_mm4)
    E_GPa = E_MPa / 1000.0

    stiffness_dominated = E_GPa > _MAX_REALISTIC_STIFFNESS_GPA

    if stiffness_dominated:
        warnings.append(
            f"Required stiffness ({E_GPa:.0f} GPa) exceeds the maximum "
            f"achievable by any engineering material "
            f"(cap: {_MAX_REALISTIC_STIFFNESS_GPA:.0f} GPa). "
            "This means the geometry — not the material — is the limiting "
            "factor. Consider: increasing wall thickness, shortening the "
            "span, or adding ribbing. "
            f"Stiffness requirement capped at {_MAX_REALISTIC_STIFFNESS_GPA} GPa "
            "and stiffness_dominated=True is set for the ML model."
        )
        E_GPa = _MAX_REALISTIC_STIFFNESS_GPA

    derivation = {
        "formula": "cantilever  E = F·L³ / (3·δ·I)",
        "F_N": F,
        "L_mm": span_mm,
        "deflection_category": load["deflection_category"],
        "deflection_divisor": div,
        "delta_limit_mm": round(delta_limit_mm, 4),
        "b_mm": b_mm,
        "t_med_mm": t_med_mm,
        "I_mm4": round(I_mm4, 4),
        "E_required_GPa": round(E_GPa, 3),
        "stiffness_dominated": stiffness_dominated,
    }

    return E_GPa, stiffness_dominated, derivation


# ===========================================================================
# Euler buckling check (compression only)
# ===========================================================================

def _check_euler_buckling(
    load: dict,
    t_med_mm: float,
    span_mm: float,
    bbox: tuple,
) -> tuple[bool, Optional[str]]:
    """
    Check whether the part is at risk of Euler column buckling.

    Uses the most conservative effective length factor K=1 (pinned-pinned).
    The applied load is compared against the critical buckling load for
    a rectangular column with cross-section (t_med × mid_bbox).

    Returns (buckling_risk_bool, warning_string_or_None).
    """
    F_applied = load["mag_max_N"]
    if F_applied <= 0:
        return False, None

    dims  = sorted(bbox)            # [min, mid, max]
    b_col = dims[1]                 # column width
    t_col = t_med_mm                # column thickness

    # I for the weaker axis of a rectangle = t³·b/12 (t < b)
    t_weak = min(t_col, b_col)
    b_weak = max(t_col, b_col)
    I_weak = (b_weak * t_weak**3) / 12.0

    # Use steel E as reference (if the material can resist buckling with
    # E=200 GPa, any higher-E material also will; if it can't, flag it)
    # This is a geometry check, not material-specific.
    E_ref_MPa = 200_000.0
    K         = 1.0           # pinned-pinned (conservative)
    L_eff     = K * span_mm

    if L_eff < 1e-6:
        return False, None

    P_cr = (math.pi**2 * E_ref_MPa * I_weak) / (L_eff**2)

    if F_applied >= P_cr:
        return True, (
            f"Euler buckling risk detected: applied load {F_applied:.0f} N "
            f">= critical buckling load {P_cr:.0f} N "
            f"(steel reference, K=1 pinned-pinned, "
            f"L_eff={L_eff:.0f} mm, I={I_weak:.1f} mm⁴). "
            "Consider increasing cross-section, adding lateral support, "
            "or shortening the unsupported length."
        )

    return False, None


# ===========================================================================
# Density budget
# ===========================================================================

def _estimate_density_budget(load: dict, geo: dict) -> float:
    """
    Estimate a maximum allowable density for the material.

    This is a soft constraint — the ML model uses it to trade off
    eco-score against weight. We derive it from the load type:

    - 'static' + 'structural': density not critical -> 10,000 kg/m³ (no limit)
    - 'cyclic' (fatigue):      weight matters for dynamic loads -> 5,000
    - general mechanical:      lightweight preferred -> 4,500
    - 'thermal' only:          density irrelevant -> 10,000

    These are order-of-magnitude guidance values, not hard limits.
    The ML model interprets them as relative weights in the objective.
    """
    load_type = load["load_type"]

    if load_type == "thermal":
        return 10_000.0

    if load_type == "static":
        return 8_000.0

    if load_type in ("cyclic", "impact"):
        # Fatigue and impact loads penalise heavy parts (inertial loads)
        return 4_500.0

    return 7_000.0  # default