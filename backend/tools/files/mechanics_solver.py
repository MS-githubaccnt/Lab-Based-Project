"""
mechanics_solver.py
===================
Closed-form structural mechanics for each ArchetypeKind.

Each solve_* function:
  - Takes a StructuralPrimitive (geometry already set) and load parameters.
  - Writes results back into the primitive in-place.
  - Returns the same primitive (for chaining).

All formulas are from:
  - Roark's Formulas for Stress and Strain (8th ed.)
  - Timoshenko & Gere, Theory of Elastic Stability (2nd ed.)
  - Beer & Johnston, Mechanics of Materials (7th ed.)

Unit conventions
----------------
  Force       : N
  Length      : mm
  Stress      : MPa  (= N/mm²)
  Young's mod : GPa  (converted to MPa internally where needed)
  Moment      : N·mm
  Second mom  : mm⁴
"""
from __future__ import annotations

import math
import logging
from typing import Callable

try:
    from .structural_primitives import ArchetypeKind, CrossSection, StructuralPrimitive
except ImportError:  # pragma: no cover - supports direct script execution in tools/files
    from structural_primitives import ArchetypeKind, CrossSection, StructuralPrimitive

logger = logging.getLogger("feature-translation")

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_MATERIAL_E_GPA     = 500.0   # diamond ~ 1000, practical cap for metals/composites
_MIN_AREA_MM2           = 0.01
_MIN_I_MM4              = 1e-9

# Deflection limits L / divisor (Roark Table 2.8 / building codes)
_DEFLECTION_DIVISORS: dict[str, float] = {
    "precision":   500.0,
    "structural":  300.0,
    "mechanical":  150.0,
    "flexible":     30.0,
}
_DEFAULT_DIV_CATEGORY = "mechanical"

# Effective length factors K for Euler buckling (Timoshenko)
_K_FACTOR: dict[str, float] = {
    "cantilever":       2.0,   # fixed-free  (worst case)
    "simply_supported": 1.0,   # pinned-pinned
    "fixed_fixed":      0.5,   # fixed-fixed (best case)
}

# Plate bending coefficients (Roark Table 11.4, simply-supported rectangular plate,
# uniform load).  a = short side, b = long side.
# Coefficients α (stress) and β (deflection) at plate centre for a/b = 1.0
# (square plate; most conservative):
#   σ_max = α · q · a² / t²
#   δ_max = β · q · a⁴ / (E · t³)
_PLATE_ALPHA = 0.2874   # Roark Table 11.4, a/b=1, simply-supported
_PLATE_BETA  = 0.01309


# ── Dispatcher ────────────────────────────────────────────────────────────────

_SOLVER_MAP: dict[ArchetypeKind, Callable] = {}


def solve_primitive(
    prim: StructuralPrimitive,
    load: dict,
    deflection_category: str,
    safety_factor: float,
    warnings: list[str],
) -> StructuralPrimitive:
    """
    Dispatch to the correct solver for prim.kind and populate results.
    """
    div_cat = deflection_category if deflection_category in _DEFLECTION_DIVISORS \
              else _DEFAULT_DIV_CATEGORY
    div = _DEFLECTION_DIVISORS[div_cat]

    solver = _SOLVER_MAP.get(prim.kind)
    if solver is None:
        warnings.append(
            f"No solver for archetype '{prim.kind.value}'. "
            "Using cantilever_beam as fallback."
        )
        solver = _solve_cantilever_beam

    solver(prim, load, div, safety_factor, warnings)
    return prim


def _register(kind: ArchetypeKind):
    def decorator(fn):
        _SOLVER_MAP[kind] = fn
        return fn
    return decorator


# ── Slender Column ─────────────────────────────────────────────────────────────

@_register(ArchetypeKind.SLENDER_COLUMN)
def _solve_slender_column(
    prim: StructuralPrimitive,
    load: dict,
    deflection_divisor: float,
    SF: float,
    warnings: list[str],
) -> None:
    """
    Column under axial compression (+ optional eccentric bending).

    1. Axial compressive stress  σ_axial = F / A
    2. Euler critical load       P_cr = π²·E·I / (K·L)²
       → Required E from P_cr ≥ F·SF:  E_req = F·SF·(K·L)² / (π²·I)
    3. Slenderness check         λ = K·L / r_gyr
       If λ > 200 → severe buckling warning.
    4. Eccentric bending (if load is not purely axial):
       Added stress from eccentricity = SF * (F * e) / Z
       where e is estimated as 0.1 * t_med (5% imperfection model).

    References: Timoshenko & Gere §1.2–§1.4; Beer & Johnston §10.3
    """
    F   = prim.applied_force_N
    L   = prim.effective_length_mm
    cs  = prim.cross_section
    K   = _K_FACTOR.get(prim.support_condition, 2.0)

    A   = max(cs.area_mm2, _MIN_AREA_MM2)
    I   = max(cs.I_mm4, _MIN_I_MM4)
    t   = cs.thickness_mm
    b   = cs.width_mm

    # Radius of gyration
    r_gyr = math.sqrt(I / A) if A > 0 else 1e-6

    # ── Axial stress
    sigma_axial = F / A  # MPa

    # ── Eccentricity (geometric imperfection model L/500 or t/10, take max)
    e_mm = max(L / 500.0, t / 10.0)
    Z    = cs.Z_mm3
    sigma_bend = (F * e_mm / Z) if Z > 1e-9 else 0.0

    sigma_total = sigma_axial + sigma_bend

    # ── Euler buckling (using reference E=200 GPa to find P_cr geometry)
    L_eff = K * L
    if L_eff < 1e-6:
        prim.buckling_risk = False
        prim.critical_buckling_load_N = float("inf")
    else:
        E_ref_MPa = 200_000.0   # steel reference for geometry check
        P_cr_ref  = (math.pi**2 * E_ref_MPa * I) / (L_eff**2)
        prim.buckling_risk = F * SF >= P_cr_ref
        prim.critical_buckling_load_N = P_cr_ref

        # Required E so that P_cr ≥ F·SF
        E_req_MPa = (F * SF * L_eff**2) / (math.pi**2 * I)
        E_req_GPa = E_req_MPa / 1000.0

    # ── Slenderness
    lam = L_eff / r_gyr if r_gyr > 1e-9 else float("inf")
    if lam > 200:
        warnings.append(
            f"Slender column: slenderness ratio λ = {lam:.0f} > 200 "
            "(Euler buckling regime). Consider increasing cross-section "
            "or shortening unsupported length."
        )

    prim.stress_MPa           = round(sigma_total, 4)
    prim.required_strength_MPa = round(sigma_total * SF, 4)
    prim.required_E_GPa       = min(round(E_req_GPa, 4), _MAX_MATERIAL_E_GPA)
    prim.governing_formula    = "axial: σ=F/A  +  eccentric: σ_b=F·e/Z  |  Euler: E_req=F·SF·(K·L)²/(π²·I)"
    prim.deflection_mm        = 0.0   # columns: buckling governs, not deflection
    prim.deflection_limit_mm  = L / deflection_divisor

    prim.derivation = {
        "F_N":              F,
        "L_mm":             L,
        "K":                K,
        "L_eff_mm":         round(L_eff, 2),
        "A_mm2":            round(A, 4),
        "I_mm4":            round(I, 4),
        "r_gyr_mm":         round(r_gyr, 4),
        "slenderness_lambda": round(lam, 1),
        "e_mm (imperfection)": round(e_mm, 4),
        "sigma_axial_MPa":  round(sigma_axial, 4),
        "sigma_bending_MPa": round(sigma_bend, 4),
        "sigma_total_MPa":  round(sigma_total, 4),
        "P_cr_N (steel ref)": round(prim.critical_buckling_load_N, 2),
        "required_E_GPa":   prim.required_E_GPa,
        "safety_factor":    SF,
    }


# ── Cantilever Beam ────────────────────────────────────────────────────────────

@_register(ArchetypeKind.CANTILEVER_BEAM)
def _solve_cantilever_beam(
    prim: StructuralPrimitive,
    load: dict,
    deflection_divisor: float,
    SF: float,
    warnings: list[str],
) -> None:
    """
    Cantilever beam under tip point load.

    σ_max   = M / Z = F·L / Z            (Roark Table 3.1a)
    δ_tip   = F·L³ / (3·E·I)             (Roark Table 3.1a)
    E_req   from δ_tip ≤ L / divisor

    References: Roark Table 3.1; Beer & Johnston §9.7
    """
    F   = prim.applied_force_N
    L   = prim.effective_length_mm
    cs  = prim.cross_section
    I   = max(cs.I_mm4, _MIN_I_MM4)
    Z   = cs.Z_mm3

    delta_lim = L / deflection_divisor

    sigma = (F * L / Z) if Z > 1e-9 else 0.0

    # E from deflection limit: FL³/(3EI) ≤ L/div → E ≥ FL²·div/(3I)
    if I > _MIN_I_MM4 and delta_lim > 1e-9:
        E_req_MPa = (F * L**2 * deflection_divisor) / (3.0 * I)
    else:
        E_req_MPa = 0.0
    E_req_GPa = E_req_MPa / 1000.0

    # Actual deflection at design E (use mid-point from strength & stiffness
    # requirements — report at E = max(E_req, 10 GPa floor))
    E_check_MPa = max(E_req_MPa, 10_000.0)
    delta_actual = (F * L**3) / (3.0 * E_check_MPa * I)

    prim.stress_MPa            = round(sigma, 4)
    prim.required_strength_MPa = round(sigma * SF, 4)
    prim.required_E_GPa        = min(round(E_req_GPa, 4), _MAX_MATERIAL_E_GPA)
    prim.deflection_mm         = round(delta_actual, 4)
    prim.deflection_limit_mm   = round(delta_lim, 4)
    prim.governing_formula     = "cantilever: σ=FL/Z  |  δ=FL³/(3EI)  →  E≥FL²·div/(3I)"

    prim.derivation = {
        "F_N":             F,
        "L_mm":            L,
        "I_mm4":           round(I, 4),
        "Z_mm3":           round(Z, 4),
        "sigma_MPa":       round(sigma, 4),
        "delta_lim_mm":    round(delta_lim, 4),
        "E_req_GPa":       round(E_req_GPa, 4),
        "safety_factor":   SF,
        "deflection_div":  deflection_divisor,
    }


# ── Simply-Supported Beam ──────────────────────────────────────────────────────

@_register(ArchetypeKind.SIMPLY_SUPPORTED_BEAM)
def _solve_simply_supported_beam(
    prim: StructuralPrimitive,
    load: dict,
    deflection_divisor: float,
    SF: float,
    warnings: list[str],
) -> None:
    """
    Simply-supported beam under central point load.

    σ_max = F·L / (4·Z)                  (Roark Table 3.1b)
    δ_max = F·L³ / (48·E·I)              (Roark Table 3.1b)
    E_req from δ_max ≤ L / divisor

    References: Roark Table 3.1b
    """
    F   = prim.applied_force_N
    L   = prim.effective_length_mm
    cs  = prim.cross_section
    I   = max(cs.I_mm4, _MIN_I_MM4)
    Z   = cs.Z_mm3

    delta_lim = L / deflection_divisor

    sigma = (F * L / (4.0 * Z)) if Z > 1e-9 else 0.0

    # E from deflection: FL³/(48EI) ≤ L/div → E ≥ FL²·div/(48I)
    if I > _MIN_I_MM4 and delta_lim > 1e-9:
        E_req_MPa = (F * L**2 * deflection_divisor) / (48.0 * I)
    else:
        E_req_MPa = 0.0
    E_req_GPa = E_req_MPa / 1000.0

    E_check_MPa = max(E_req_MPa, 10_000.0)
    delta_actual = (F * L**3) / (48.0 * E_check_MPa * I)

    prim.stress_MPa            = round(sigma, 4)
    prim.required_strength_MPa = round(sigma * SF, 4)
    prim.required_E_GPa        = min(round(E_req_GPa, 4), _MAX_MATERIAL_E_GPA)
    prim.deflection_mm         = round(delta_actual, 4)
    prim.deflection_limit_mm   = round(delta_lim, 4)
    prim.governing_formula     = "SS beam: σ=FL/(4Z)  |  δ=FL³/(48EI)  →  E≥FL²·div/(48I)"

    prim.derivation = {
        "F_N":           F,
        "L_mm":          L,
        "I_mm4":         round(I, 4),
        "Z_mm3":         round(Z, 4),
        "sigma_MPa":     round(sigma, 4),
        "delta_lim_mm":  round(delta_lim, 4),
        "E_req_GPa":     round(E_req_GPa, 4),
        "safety_factor": SF,
        "deflection_div": deflection_divisor,
    }


# ── Flat Plate ─────────────────────────────────────────────────────────────────

@_register(ArchetypeKind.FLAT_PLATE)
def _solve_flat_plate(
    prim: StructuralPrimitive,
    load: dict,
    deflection_divisor: float,
    SF: float,
    warnings: list[str],
) -> None:
    """
    Simply-supported rectangular plate under uniform distributed load.

    σ_max = α · q · a² / t²              (Roark Table 11.4)
    δ_max = β · q · a⁴ / (E · t³)       (Roark Table 11.4)

    where:
      q  = pressure load [N/mm²]  = F / plan_area
      a  = shorter in-plane span  [mm]
      t  = plate thickness        [mm]
      α  = 0.2874  (Roark, a/b=1, SS, uniform q)
      β  = 0.01309 (Roark, a/b=1, SS, uniform q)

    E_req from δ_max ≤ a / divisor:
      E_req = β · q · a³ · divisor / t³

    References: Roark Table 11.4; Timoshenko & Woinowsky-Krieger §5
    """
    F   = prim.applied_force_N
    cs  = prim.cross_section
    t   = cs.thickness_mm
    a   = prim.effective_length_mm   # shorter span (set in detector)
    b   = prim.bbox.span              # longer span

    # Plan area for pressure
    plan_area = max(a * b, _MIN_AREA_MM2)
    q = F / plan_area   # N/mm²

    # Account for a/b ratio — Roark coefficients for square plate are
    # conservative (maximum stress); use them directly.
    if b > 0 and a / b < 0.5:
        warnings.append(
            f"Plate aspect ratio a/b = {a/b:.2f} < 0.5. "
            "Roark square-plate coefficients (α=0.2874, β=0.01309) are "
            "conservative; actual stress will be lower."
        )

    sigma = _PLATE_ALPHA * q * a**2 / (t**2) if t > 1e-6 else 0.0

    delta_lim = a / deflection_divisor
    if t > 1e-6 and delta_lim > 1e-9:
        E_req_MPa = (_PLATE_BETA * q * a**3 * deflection_divisor) / (t**3)
    else:
        E_req_MPa = 0.0
    E_req_GPa = E_req_MPa / 1000.0

    E_check_MPa = max(E_req_MPa, 10_000.0)
    delta_actual = (_PLATE_BETA * q * a**4) / (E_check_MPa * t**3) if t > 1e-6 else 0.0

    prim.stress_MPa            = round(sigma, 4)
    prim.required_strength_MPa = round(sigma * SF, 4)
    prim.required_E_GPa        = min(round(E_req_GPa, 4), _MAX_MATERIAL_E_GPA)
    prim.deflection_mm         = round(delta_actual, 4)
    prim.deflection_limit_mm   = round(delta_lim, 4)
    prim.governing_formula     = (
        "flat plate (Roark 11.4): σ=α·q·a²/t²  |  δ=β·q·a⁴/(E·t³)"
        "  →  E≥β·q·a³·div/t³"
    )

    prim.derivation = {
        "F_N":             F,
        "q_N_mm2":         round(q, 6),
        "a_mm (short span)": round(a, 2),
        "b_mm (long span)": round(b, 2),
        "t_mm":            round(t, 4),
        "alpha_Roark":     _PLATE_ALPHA,
        "beta_Roark":      _PLATE_BETA,
        "sigma_MPa":       round(sigma, 4),
        "delta_lim_mm":    round(delta_lim, 4),
        "E_req_GPa":       round(E_req_GPa, 4),
        "safety_factor":   SF,
        "deflection_div":  deflection_divisor,
    }


# ── Cantilever Plate ───────────────────────────────────────────────────────────

@_register(ArchetypeKind.CANTILEVER_PLATE)
def _solve_cantilever_plate(
    prim: StructuralPrimitive,
    load: dict,
    deflection_divisor: float,
    SF: float,
    warnings: list[str],
) -> None:
    """
    Cantilever plate (shelf / bracket) under uniform load over its surface.

    Treats as a wide cantilever beam: per-unit-width strip approach.
    Width b cancels in stress formula; thickness t governs.

    Per unit width strip (b=1 mm):
      M_max = q_line · L² / 2           (q_line = q · b = q [N/mm])
      σ_max = M / (t²/6) = 3·q·L² / t²
      δ_tip = q·L⁴ / (8·E·I_1)         where I_1 = t³/12 per unit width

    q = F / plan_area [N/mm²]

    References: Roark Table 3.1; Timoshenko §5.12
    """
    F   = prim.applied_force_N
    cs  = prim.cross_section
    t   = cs.thickness_mm
    L   = prim.effective_length_mm
    b   = cs.width_mm   # plate width (breadth perpendicular to cantilever)

    plan_area = max(L * b, _MIN_AREA_MM2)
    q = F / plan_area   # N/mm²

    # Per-unit-width
    I_1 = (t**3) / 12.0   # mm⁴/mm
    Z_1 = (t**2) / 6.0    # mm³/mm

    sigma = (3.0 * q * L**2) / (t**2) if t > 1e-6 else 0.0

    delta_lim = L / deflection_divisor
    if I_1 > 1e-12 and delta_lim > 1e-9:
        E_req_MPa = (q * L**3 * deflection_divisor) / (8.0 * I_1)
    else:
        E_req_MPa = 0.0
    E_req_GPa = E_req_MPa / 1000.0

    E_check_MPa = max(E_req_MPa, 10_000.0)
    delta_actual = (q * L**4) / (8.0 * E_check_MPa * I_1) if I_1 > 1e-12 else 0.0

    prim.stress_MPa            = round(sigma, 4)
    prim.required_strength_MPa = round(sigma * SF, 4)
    prim.required_E_GPa        = min(round(E_req_GPa, 4), _MAX_MATERIAL_E_GPA)
    prim.deflection_mm         = round(delta_actual, 4)
    prim.deflection_limit_mm   = round(delta_lim, 4)
    prim.governing_formula     = (
        "cantilever plate: σ=3·q·L²/t²  |  δ=q·L⁴/(8·E·I₁)"
        "  →  E≥q·L³·div/(8·I₁)"
    )

    prim.derivation = {
        "F_N":           F,
        "q_N_mm2":       round(q, 6),
        "L_mm":          round(L, 2),
        "b_mm":          round(b, 2),
        "t_mm":          round(t, 4),
        "I_1_per_mm":    round(I_1, 6),
        "sigma_MPa":     round(sigma, 4),
        "delta_lim_mm":  round(delta_lim, 4),
        "E_req_GPa":     round(E_req_GPa, 4),
        "safety_factor": SF,
    }


# ── Thin-Walled Shell ──────────────────────────────────────────────────────────

@_register(ArchetypeKind.THIN_WALLED_SHELL)
def _solve_thin_walled_shell(
    prim: StructuralPrimitive,
    load: dict,
    deflection_divisor: float,
    SF: float,
    warnings: list[str],
) -> None:
    """
    Thin-walled cylindrical shell under internal/external pressure or radial load.

    Hoop stress:      σ_θ = p · r / t              (Lamé thin-wall)
    Meridional stress: σ_z = p · r / (2t)
    Governing:         σ_max = σ_θ

    Radial deflection: δ_r = p · r² / (E · t)
    E_req from δ_r ≤ r / divisor:  E_req = p · r · divisor / t

    Pressure estimate: p = F / (2π · r · L)   (lateral line load on cylinder)

    References: Roark Table 13.1; Beer & Johnston §8.2
    """
    F   = prim.applied_force_N
    cs  = prim.cross_section
    t   = cs.thickness_mm
    L   = prim.effective_length_mm

    # Estimate radius from bbox mid-dimension
    dims = prim.bbox.sorted_dims
    r = dims[1] / 2.0   # mid dim / 2

    if r < 1e-6:
        prim.warnings.append("Shell radius near zero. Results unreliable.")
        r = 1.0

    # Radial pressure equivalent
    lateral_area = max(2.0 * math.pi * r * L, _MIN_AREA_MM2)
    p = F / lateral_area   # N/mm²

    sigma_hoop = (p * r / t) if t > 1e-6 else 0.0
    sigma_merid = sigma_hoop / 2.0

    delta_lim = r / deflection_divisor
    if t > 1e-6 and delta_lim > 1e-9:
        E_req_MPa = (p * r * deflection_divisor) / t
    else:
        E_req_MPa = 0.0
    E_req_GPa = E_req_MPa / 1000.0

    E_check_MPa = max(E_req_MPa, 10_000.0)
    delta_r = (p * r**2) / (E_check_MPa * t) if t > 1e-6 else 0.0

    prim.stress_MPa            = round(sigma_hoop, 4)
    prim.required_strength_MPa = round(sigma_hoop * SF, 4)
    prim.required_E_GPa        = min(round(E_req_GPa, 4), _MAX_MATERIAL_E_GPA)
    prim.deflection_mm         = round(delta_r, 4)
    prim.deflection_limit_mm   = round(delta_lim, 4)
    prim.governing_formula     = (
        "thin shell (Lamé): σ_θ=p·r/t  |  δ_r=p·r²/(E·t)"
        "  →  E≥p·r·div/t"
    )

    prim.derivation = {
        "F_N":           F,
        "r_mm":          round(r, 2),
        "t_mm":          round(t, 4),
        "L_mm":          round(L, 2),
        "p_N_mm2":       round(p, 8),
        "sigma_hoop_MPa": round(sigma_hoop, 4),
        "sigma_meridional_MPa": round(sigma_merid, 4),
        "delta_lim_mm":  round(delta_lim, 4),
        "E_req_GPa":     round(E_req_GPa, 4),
        "safety_factor": SF,
    }


# ── Solid Block ────────────────────────────────────────────────────────────────

@_register(ArchetypeKind.SOLID_BLOCK)
def _solve_solid_block(
    prim: StructuralPrimitive,
    load: dict,
    deflection_divisor: float,
    SF: float,
    warnings: list[str],
) -> None:
    """
    Solid block under bearing / crushing load.

    σ_bearing = F / A_contact
    E_req: compressive deflection δ = F·L/(A·E) ≤ L/divisor
      → E_req = F · divisor / A

    References: Beer & Johnston §2.2
    """
    F  = prim.applied_force_N
    cs = prim.cross_section
    A  = max(cs.area_mm2, _MIN_AREA_MM2)
    L  = prim.effective_length_mm

    sigma = F / A

    E_req_MPa = (F * deflection_divisor) / A if A > _MIN_AREA_MM2 else 0.0
    E_req_GPa = E_req_MPa / 1000.0

    E_check_MPa = max(E_req_MPa, 10_000.0)
    delta = (F * L) / (A * E_check_MPa)
    delta_lim = L / deflection_divisor

    prim.stress_MPa            = round(sigma, 4)
    prim.required_strength_MPa = round(sigma * SF, 4)
    prim.required_E_GPa        = min(round(E_req_GPa, 4), _MAX_MATERIAL_E_GPA)
    prim.deflection_mm         = round(delta, 4)
    prim.deflection_limit_mm   = round(delta_lim, 4)
    prim.governing_formula     = "bearing: σ=F/A  |  δ=FL/(AE)  →  E≥F·div/A"

    prim.derivation = {
        "F_N":           F,
        "A_mm2":         round(A, 4),
        "L_mm":          round(L, 2),
        "sigma_MPa":     round(sigma, 4),
        "E_req_GPa":     round(E_req_GPa, 4),
        "safety_factor": SF,
    }
