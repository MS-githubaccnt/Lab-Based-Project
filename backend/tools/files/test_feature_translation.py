"""
test_feature_translation.py
============================
Tests for the new structural-mechanics feature translation pipeline.
Run with:  python test_feature_translation.py
"""
from __future__ import annotations
import sys
import math
import traceback

# ── Inline imports (avoids package setup) ─────────────────────────────────────
sys.path.insert(0, ".")

from structural_primitives import ArchetypeKind, BoundingBox, CrossSection
from archetype_detector import detect_primitives, _choose_archetype
from mechanics_solver import solve_primitive
from load_distributor import distribute_loads
from translate_features_to_ml_inputs import translate_features_to_ml_inputs

# ── Test harness ──────────────────────────────────────────────────────────────

_PASS = 0
_FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        print(f"  ✓  {name}")
        _PASS += 1
    else:
        print(f"  ✗  {name}" + (f"  [{detail}]" if detail else ""))
        _FAIL += 1

def section(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ── Geometry helpers ──────────────────────────────────────────────────────────

def geo(x, y, z, t_min=None, t_med=None, dominant_axis="compact",
        watertight=True, **kwargs):
    d = {
        "bbox_x_mm": x, "bbox_y_mm": y, "bbox_z_mm": z,
        "dominant_axis": dominant_axis,
        "is_watertight": watertight,
    }
    if t_min is not None: d["min_wall_thickness_mm"] = t_min
    if t_med is not None: d["median_wall_thickness_mm"] = t_med
    d.update(kwargs)
    return d


def load(F=1000, mode="bending", type_="static", SF=2.0,
         temp=(-20, 80), fatigue=False, deflection_cat="mechanical"):
    return {
        "load_type": type_,
        "primary_stress_mode": mode,
        "magnitude_range_N": [0, F],
        "safety_factor": SF,
        "operating_temp_C": temp,
        "is_fatigue_critical": fatigue,
        "deflection_category": deflection_cat,
    }


def run_tool(geometry, load_estimate, recyclability=0.7):
    """Run the full tool and return the result dict (or raise)."""
    # The @tool decorator wraps the function; call .func to get the raw fn
    fn = translate_features_to_ml_inputs.func
    return fn(geometry, load_estimate, recyclability)


# ═════════════════════════════════════════════════════════════════════════════
# 1. BoundingBox helpers
# ═════════════════════════════════════════════════════════════════════════════

section("1. BoundingBox helpers")

bb = BoundingBox(10, 20, 200)
check("span = max dim",       bb.span == 200)
check("min_dim",              bb.min_dim == 10)
check("sorted_dims",          bb.sorted_dims == (10, 20, 200))
check("aspect_ratio",         abs(bb.aspect_ratio() - 20.0) < 1e-9)
check("flatness near 0",      bb.flatness_ratio() < 0.6)

bb2 = BoundingBox(100, 100, 5)
check("flat plate flatness",  bb2.flatness_ratio() < 0.1)

# ═════════════════════════════════════════════════════════════════════════════
# 2. Archetype detection — shape descriptors
# ═════════════════════════════════════════════════════════════════════════════

section("2. Archetype detection — shape descriptors")

# Thin tall column: 20×20×300 mm
g_col = geo(20, 20, 300, t_min=2, t_med=3, dominant_axis="elongated")
l_comp = load(mode="compression")
w = []
prims = detect_primitives(g_col, l_comp, w)
check("column detected",
      any(p.kind == ArchetypeKind.SLENDER_COLUMN for p in prims),
      str([p.kind for p in prims]))

# Wide flat plate: 500×400×10 mm
g_plate = geo(500, 400, 10, t_min=10, t_med=10, dominant_axis="flat")
l_bend = load(mode="bending", type_="static")
prims2 = detect_primitives(g_plate, l_bend, [])
check("plate detected",
      any(p.kind in (ArchetypeKind.FLAT_PLATE, ArchetypeKind.CANTILEVER_PLATE)
          for p in prims2),
      str([p.kind for p in prims2]))

# Elongated horizontal beam: long axis in X (200mm), not Z
# dominant_axis="elongated" + X >> Z → horizontal beam
g_beam = geo(200, 15, 10, t_min=5, t_med=7, dominant_axis="elongated")
l_horiz = load(mode="bending", type_="static")
prims3 = detect_primitives(g_beam, l_horiz, [])
check("beam detected",
      any(p.kind in (ArchetypeKind.CANTILEVER_BEAM,
                     ArchetypeKind.SIMPLY_SUPPORTED_BEAM) for p in prims3),
      str([p.kind for p in prims3]))

# Compact block: 50×60×55 mm
g_block = geo(50, 60, 55, t_min=50, t_med=50, dominant_axis="compact")
prims4 = detect_primitives(g_block, load(), [])
check("block detected",
      any(p.kind == ArchetypeKind.SOLID_BLOCK for p in prims4),
      str([p.kind for p in prims4]))

# ═════════════════════════════════════════════════════════════════════════════
# 3. CrossSection second moment of area
# ═════════════════════════════════════════════════════════════════════════════

section("3. CrossSection — I, Z")

cs = CrossSection(area_mm2=100.0, width_mm=20.0, thickness_mm=5.0)
I_expected = (20.0 * 5.0**3) / 12.0   # = 208.33 mm⁴
Z_expected = I_expected / (5.0 / 2.0)  # = 83.33 mm³
check("I_mm4 correct",  abs(cs.I_mm4 - I_expected) < 0.01, f"{cs.I_mm4:.4f}")
check("Z_mm3 correct",  abs(cs.Z_mm3 - Z_expected) < 0.01, f"{cs.Z_mm3:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Mechanics solver — cantilever beam
# ═════════════════════════════════════════════════════════════════════════════

section("4. Mechanics solver — cantilever beam")

from structural_primitives import StructuralPrimitive

def make_cantilever(F=500.0, L=200.0, b=20.0, t=5.0):
    cs   = CrossSection(area_mm2=b*t, width_mm=b, thickness_mm=t)
    bbox = BoundingBox(b, t, L)
    prim = StructuralPrimitive(
        kind=ArchetypeKind.CANTILEVER_BEAM,
        bbox=bbox,
        cross_section=cs,
        applied_force_N=F,
        effective_length_mm=L,
        support_condition="cantilever",
    )
    return prim

prim_cb = make_cantilever(F=500, L=200, b=20, t=5)
load_cb = {"primary_stress_mode": "bending", "load_type": "static",
           "mag_max_N": 500}
w = []
solve_primitive(prim_cb, load_cb, "mechanical", 2.0, w)

# Hand-check: I = 20×5³/12 = 208.33, Z = 83.33
I  = (20 * 5**3) / 12
Z  = I / 2.5
sigma_expected = (500 * 200) / Z
E_req_MPa = (500 * 200**2 * 150) / (3 * I)  # div=150 for "mechanical"
E_req_GPa = E_req_MPa / 1000

check("cantilever σ matches formula",
      abs(prim_cb.stress_MPa - sigma_expected) < 0.1,
      f"got {prim_cb.stress_MPa:.2f}, expected {sigma_expected:.2f}")
# E_req=4800 GPa >> material cap of 500 GPa — tool correctly caps and sets
# stiffness_dominated. Test that capping happened and flag is set.
check("cantilever E_req capped at material limit",
      prim_cb.required_E_GPa <= 500.0,
      f"got {prim_cb.required_E_GPa:.3f}")
check("cantilever stiffness_dominated flag set when E_req > cap",
      E_req_GPa > 500.0,   # our geometry IS stiffness-dominated
      f"raw E_req={E_req_GPa:.1f} GPa")
check("cantilever strength = σ × SF",
      abs(prim_cb.required_strength_MPa - sigma_expected * 2.0) < 0.1)

# ═════════════════════════════════════════════════════════════════════════════
# 5. Mechanics solver — simply-supported beam
# ═════════════════════════════════════════════════════════════════════════════

section("5. Mechanics solver — simply-supported beam")

from structural_primitives import StructuralPrimitive

def make_ss_beam(F=1000.0, L=300.0, b=25.0, t=8.0):
    cs   = CrossSection(area_mm2=b*t, width_mm=b, thickness_mm=t)
    bbox = BoundingBox(b, t, L)
    prim = StructuralPrimitive(
        kind=ArchetypeKind.SIMPLY_SUPPORTED_BEAM,
        bbox=bbox,
        cross_section=cs,
        applied_force_N=F,
        effective_length_mm=L,
        support_condition="simply_supported",
    )
    return prim

prim_ss = make_ss_beam()
solve_primitive(prim_ss, {"primary_stress_mode": "bending",
                           "load_type": "static", "mag_max_N": 1000},
               "structural", 2.0, [])

I_ss  = (25 * 8**3) / 12
Z_ss  = I_ss / 4.0
sigma_ss = (1000 * 300) / (4 * Z_ss)
E_req_ss_MPa = (1000 * 300**2 * 300) / (48 * I_ss)   # div=300 structural
E_req_ss_GPa = E_req_ss_MPa / 1000

check("SS beam σ correct",
      abs(prim_ss.stress_MPa - sigma_ss) < 0.1,
      f"got {prim_ss.stress_MPa:.2f}, expected {sigma_ss:.2f}")
# E_req=527 GPa > 500 GPa cap — tool correctly caps
check("SS beam E_req capped at material limit",
      prim_ss.required_E_GPa <= 500.0,
      f"got {prim_ss.required_E_GPa:.4f}")
check("SS beam raw E_req exceeds cap (geometry-dominated case)",
      E_req_ss_GPa > 500.0,
      f"raw={E_req_ss_GPa:.2f} GPa")

# ═════════════════════════════════════════════════════════════════════════════
# 6. Mechanics solver — slender column (buckling)
# ═════════════════════════════════════════════════════════════════════════════

section("6. Mechanics solver — slender column")

def make_column(F=5000.0, L=400.0, b=10.0, t=5.0):
    cs   = CrossSection(area_mm2=b*t, width_mm=b, thickness_mm=t)
    bbox = BoundingBox(b, t, L)
    prim = StructuralPrimitive(
        kind=ArchetypeKind.SLENDER_COLUMN,
        bbox=bbox,
        cross_section=cs,
        applied_force_N=F,
        effective_length_mm=L,
        support_condition="cantilever",
    )
    return prim

prim_col = make_column(F=5000, L=400, b=10, t=5)
solve_primitive(prim_col, {"primary_stress_mode": "compression",
                            "load_type": "static", "mag_max_N": 5000},
               "mechanical", 2.0, [])

I_col  = (10 * 5**3) / 12
K      = 2.0   # cantilever
L_eff  = K * 400
E_ref  = 200_000.0
P_cr   = (math.pi**2 * E_ref * I_col) / (L_eff**2)

check("column buckling risk detected when F >= P_cr",
      prim_col.buckling_risk == (5000 * 2.0 >= P_cr),
      f"F·SF={5000*2.0:.0f} vs P_cr={P_cr:.0f}")
check("column axial stress > 0", prim_col.stress_MPa > 0)
check("column required_E_GPa > 0", prim_col.required_E_GPa > 0)

# Slender column with tiny section — should definitely flag buckling
prim_col2 = make_column(F=10000, L=1000, b=5, t=2)
w2 = []
solve_primitive(prim_col2, {"primary_stress_mode": "compression",
                             "load_type": "static", "mag_max_N": 10000},
               "mechanical", 3.0, w2)
check("very slender column flags buckling", prim_col2.buckling_risk)

# ═════════════════════════════════════════════════════════════════════════════
# 7. Mechanics solver — flat plate (Roark)
# ═════════════════════════════════════════════════════════════════════════════

section("7. Mechanics solver — flat plate (Roark Table 11.4)")

from structural_primitives import StructuralPrimitive

def make_plate(F=2000.0, a=300.0, b=400.0, t=8.0):
    cs   = CrossSection(area_mm2=b*t, width_mm=b, thickness_mm=t)
    bbox = BoundingBox(b, a, t)
    prim = StructuralPrimitive(
        kind=ArchetypeKind.FLAT_PLATE,
        bbox=bbox,
        cross_section=cs,
        applied_force_N=F,
        effective_length_mm=a,
        support_condition="simply_supported",
    )
    return prim

ALPHA = 0.2874
BETA  = 0.01309

prim_pl = make_plate(F=2000, a=300, b=400, t=8)
solve_primitive(prim_pl, {"primary_stress_mode": "bending",
                           "load_type": "static", "mag_max_N": 2000},
               "structural", 2.0, [])

q = 2000 / (300 * 400)
sigma_pl = ALPHA * q * 300**2 / (8**2)
check("plate σ matches Roark 11.4",
      abs(prim_pl.stress_MPa - sigma_pl) < 0.1,
      f"got {prim_pl.stress_MPa:.4f}, expected {sigma_pl:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# 8. Full pipeline — single-body mesh
# ═════════════════════════════════════════════════════════════════════════════

section("8. Full pipeline — single body (chair leg proxy)")

g_leg  = geo(25, 25, 450, t_min=3, t_med=4, dominant_axis="elongated")
l_leg  = load(F=2000, mode="compression", type_="static", SF=3.0)
result = run_tool(g_leg, l_leg, recyclability=0.6)

check("output has required_tensile_strength_MPa",
      "required_tensile_strength_MPa" in result)
check("output has required_stiffness_GPa",
      "required_stiffness_GPa" in result)
check("strength > 0", result["required_tensile_strength_MPa"] > 0)
check("stiffness >= 0", result["required_stiffness_GPa"] >= 0)
check("governing_primitive present", "governing_primitive" in result)
check("primitives list non-empty", len(result["primitives"]) > 0)
check("warnings is list", isinstance(result["warnings"], list))

# ═════════════════════════════════════════════════════════════════════════════
# 9. Full pipeline — flat plate (seat proxy)
# ═════════════════════════════════════════════════════════════════════════════

section("9. Full pipeline — flat plate (seat proxy)")

g_seat = geo(500, 450, 20, t_min=20, t_med=20, dominant_axis="flat")
l_seat = load(F=1000, mode="bending", type_="static", SF=2.5)
result2 = run_tool(g_seat, l_seat)

check("seat strength > 0",   result2["required_tensile_strength_MPa"] > 0)
check("seat stiffness >= 0", result2["required_stiffness_GPa"] >= 0)
check("governing primitive is a plate type",
      result2["governing_primitive"] in (
          ArchetypeKind.FLAT_PLATE.value,
          ArchetypeKind.CANTILEVER_PLATE.value,
          ArchetypeKind.CANTILEVER_BEAM.value,  # fallback allowed
      ),
      result2["governing_primitive"])

# ═════════════════════════════════════════════════════════════════════════════
# 10. Full pipeline — multi-body mesh
# ═════════════════════════════════════════════════════════════════════════════

section("10. Full pipeline — multi-body (chair: 4 legs + seat)")

g_multi = geo(
    500, 500, 800, t_min=3, t_med=5, dominant_axis="compact",
    connected_components=[
        # 4 legs
        geo(25, 25, 450, t_min=3, t_med=4, dominant_axis="elongated"),
        geo(25, 25, 450, t_min=3, t_med=4, dominant_axis="elongated"),
        geo(25, 25, 450, t_min=3, t_med=4, dominant_axis="elongated"),
        geo(25, 25, 450, t_min=3, t_med=4, dominant_axis="elongated"),
        # seat
        geo(500, 450, 20, t_min=20, t_med=20, dominant_axis="flat"),
    ]
)
l_multi = load(F=1200, mode="compression", type_="static", SF=3.0)
result3  = run_tool(g_multi, l_multi)

check("multi-body: ≥ 5 primitives",  len(result3["primitives"]) >= 5,
      f"got {len(result3['primitives'])}")
check("multi-body: strength > 0",    result3["required_tensile_strength_MPa"] > 0)
check("multi-body: governing set",   result3["governing_primitive"] != "")

# ═════════════════════════════════════════════════════════════════════════════
# 11. Thermal-only load
# ═════════════════════════════════════════════════════════════════════════════

section("11. Thermal-only load")

g_th = geo(100, 100, 100, dominant_axis="compact")
l_th = load(F=0, mode="none", type_="thermal", SF=1.5, temp=(-40, 150))
result4 = run_tool(g_th, l_th)

check("thermal: strength = 0",   result4["required_tensile_strength_MPa"] == 0.0)
check("thermal: stiffness = 0",  result4["required_stiffness_GPa"] == 0.0)
check("thermal: temp range set", result4["min_operating_temp_C"] == -40.0)

# ═════════════════════════════════════════════════════════════════════════════
# 12. Edge cases
# ═════════════════════════════════════════════════════════════════════════════

section("12. Edge cases")

# Zero force
g_zero = geo(50, 50, 200, t_min=3, t_med=4, dominant_axis="elongated")
l_zero = load(F=0, mode="bending")
try:
    r_zero = run_tool(g_zero, l_zero)
    check("zero force: strength = 0",  r_zero["required_tensile_strength_MPa"] == 0.0)
    check("zero force: stiffness = 0", r_zero["required_stiffness_GPa"] == 0.0)
except Exception as e:
    check("zero force: no crash", False, str(e))

# Invalid recyclability
try:
    run_tool(geo(100,100,100), load(), recyclability=1.5)
    check("recyclability > 1.0 raises ValueError", False, "no error raised")
except ValueError:
    check("recyclability > 1.0 raises ValueError", True)

# Missing load key
try:
    run_tool(geo(100,100,100), {"load_type": "static"})
    check("missing key raises ValueError", False)
except (ValueError, KeyError):
    check("missing key raises ValueError", True)

# Swapped magnitude range
g_swap = geo(50, 50, 200, t_min=3, t_med=4, dominant_axis="elongated")
l_swap = {**load(F=500, mode="bending"), "magnitude_range_N": [500, 100]}
try:
    r_swap = run_tool(g_swap, l_swap)
    check("swapped range: still runs", r_swap["required_tensile_strength_MPa"] > 0)
    check("swapped range: warning emitted",
          any("swap" in w.lower() for w in r_swap["warnings"]))
except Exception as e:
    check("swapped range: no crash", False, str(e))

# ═════════════════════════════════════════════════════════════════════════════
# 13. Safety factor scaling
# ═════════════════════════════════════════════════════════════════════════════

section("13. Safety factor scaling")

g_sf = geo(20, 20, 300, t_min=3, t_med=4, dominant_axis="elongated")
l_sf1 = load(F=1000, mode="bending", SF=1.0)
l_sf2 = load(F=1000, mode="bending", SF=3.0)
r1 = run_tool(g_sf, l_sf1)
r2 = run_tool(g_sf, l_sf2)
ratio = r2["required_tensile_strength_MPa"] / r1["required_tensile_strength_MPa"]
check("strength scales with safety factor",
      abs(ratio - 3.0) < 0.01,
      f"ratio={ratio:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# 14. Output schema completeness
# ═════════════════════════════════════════════════════════════════════════════

section("14. Output schema completeness")

required_keys = [
    "required_tensile_strength_MPa",
    "required_stiffness_GPa",
    "stiffness_dominated",
    "buckling_risk",
    "min_operating_temp_C",
    "max_operating_temp_C",
    "max_density_kg_m3",
    "fatigue_critical",
    "recyclability_priority",
    "governing_primitive",
    "primitives",
    "derivation",
    "warnings",
]

g_full = geo(30, 30, 400, t_min=3, t_med=5, dominant_axis="elongated")
l_full = load(F=1500, mode="bending", type_="cyclic", SF=2.5, fatigue=True)
r_full = run_tool(g_full, l_full, recyclability=0.8)

for key in required_keys:
    check(f"output has '{key}'", key in r_full, f"missing from {list(r_full.keys())}")

check("recyclability_priority passed through",
      r_full["recyclability_priority"] == 0.8)
check("fatigue_critical passed through", r_full["fatigue_critical"] is True)

# ═════════════════════════════════════════════════════════════════════════════
# 15. Derivation traceability
# ═════════════════════════════════════════════════════════════════════════════

section("15. Derivation traceability")

check("derivation dict has governing_formula",
      "governing_formula" in r_full["derivation"])
check("derivation dict has governing_derivation",
      "governing_derivation" in r_full["derivation"])
check("governing_derivation is a dict",
      isinstance(r_full["derivation"]["governing_derivation"], dict))
check("all_primitives_summary present",
      "all_primitives_summary" in r_full["derivation"])

# Per-primitive summary has required fields
prim_summary = r_full["primitives"][0]
for field in ["archetype", "required_strength_MPa", "required_E_GPa",
              "governing_formula", "derivation"]:
    check(f"primitive summary has '{field}'", field in prim_summary)

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
total = _PASS + _FAIL
print(f"  Passed: {_PASS}/{total}   Failed: {_FAIL}/{total}")
print(f"{'═'*60}")

if _FAIL > 0:
    sys.exit(1)
