"""
archetype_detector.py
=====================
Classifies the geometry dict (output of parse_and_extract_cad_features /
trimesh analysis) into one or more StructuralPrimitive objects.

Design contract
---------------
- Zero LLM calls. Fully deterministic.
- Works on any STL — no object-type assumptions.
- Uses only geometry: bounding box(es), wall thickness, curvature metrics,
  face-normal distribution, connected-component topology.

Classification hierarchy (evaluated in order):
  1. Per connected-component if multi-body mesh.
  2. Whole-mesh fallback for single-body meshes.

Each primitive is classified by shape descriptors:
  - aspect_ratio  = span / min_dim  (elongation)
  - flatness      = min_dim / mid_dim  (plate-ness; near 0 = very flat)
  - curvature_ratio = mean_curvature * span  (shell-ness)
  - slenderness   = effective_length / r_gyration  (column buckling indicator)
"""
from __future__ import annotations

import math
import logging
from typing import Optional

try:
    from .structural_primitives import (
        ArchetypeKind,
        BoundingBox,
        CrossSection,
        StructuralPrimitive,
    )
except ImportError:  # pragma: no cover - supports direct script execution in tools/files
    from structural_primitives import (
        ArchetypeKind,
        BoundingBox,
        CrossSection,
        StructuralPrimitive,
    )

logger = logging.getLogger("feature-translation")

# ── Thresholds (all dimensionless unless noted) ───────────────────────────────
_ASPECT_ELONGATED       = 4.0    # L/t > this → beam/column candidate
_ASPECT_VERY_ELONGATED  = 12.0   # L/t > this → strong column/beam signal
_FLATNESS_PLATE         = 0.25   # min/mid < this → plate candidate
_FLATNESS_VERY_FLAT     = 0.10   # min/mid < this → strong plate signal
_CURVATURE_SHELL        = 0.05   # curvature_ratio > this → shell candidate
_SLENDERNESS_BUCKLING   = 50.0   # L/r_gyration > this → buckling risk zone


def detect_primitives(
    geometry: dict,
    load_estimate: dict,
    warnings: list[str],
) -> list[StructuralPrimitive]:
    """
    Main entry point.  Returns a list of StructuralPrimitive objects —
    one per connected component (or one for single-body meshes).

    Parameters
    ----------
    geometry :
        Enriched geometry dict from trimesh analysis.
        Required keys:
            bbox_x_mm, bbox_y_mm, bbox_z_mm
            min_wall_thickness_mm (optional)
            median_wall_thickness_mm (optional)
            dominant_axis
            is_watertight
        Optional enrichment keys (from enhanced trimesh extractor):
            connected_components : list[dict]  — per-body geometry dicts
            mean_curvature       : float       — mean absolute curvature (1/mm)
            bottom_contact_area_mm2 : float
            top_load_area_mm2    : float
            volume_mm3           : float

    load_estimate :
        Parsed load dict (output of _parse_load_estimate).

    warnings :
        Mutable list; warnings are appended in-place.

    Returns
    -------
    list[StructuralPrimitive]
        At least one primitive.
    """
    components: list[dict] = geometry.get("connected_components", [])

    if len(components) > 1:
        logger.debug("Multi-body mesh: %d components", len(components))
        primitives = []
        for i, comp in enumerate(components):
            comp = _with_parent_defaults(comp, geometry)
            p = _classify_component(comp, load_estimate, warnings, label=f"comp_{i}")
            if p is not None:
                primitives.append(p)
        if not primitives:
            warnings.append(
                "All connected components were degenerate. "
                "Falling back to whole-mesh classification."
            )
            primitives = [_classify_whole_mesh(geometry, load_estimate, warnings)]
    else:
        primitives = [_classify_whole_mesh(geometry, load_estimate, warnings)]

    return primitives


def _with_parent_defaults(component: dict, parent: dict) -> dict:
    """
    Preserve component-level geometry while filling missing measured fields from
    the whole mesh. This keeps open STL fragments from dropping into bbox-only
    thickness estimates when the full parser already measured a representative
    wall thickness.
    """
    enriched = dict(component)
    for key in (
        "min_wall_thickness_mm",
        "median_wall_thickness_mm",
        "is_watertight",
        "has_cylindrical_bores",
    ):
        if enriched.get(key) is None and parent.get(key) is not None:
            enriched[key] = parent[key]
    return enriched


# ── Component-level classifier ─────────────────────────────────────────────────

def _classify_component(
    comp: dict,
    load_estimate: dict,
    warnings: list[str],
    label: str,
) -> Optional[StructuralPrimitive]:
    """Classify one connected component."""
    try:
        return _classify_whole_mesh(comp, load_estimate, warnings)
    except Exception as exc:
        warnings.append(f"Component {label} skipped: {exc}")
        return None


# ── Whole-mesh (or per-component) classifier ──────────────────────────────────

def _classify_whole_mesh(
    geo: dict,
    load_estimate: dict,
    warnings: list[str],
) -> StructuralPrimitive:
    """
    Classify one body (whole mesh or one component) into a StructuralPrimitive.
    """
    bbox = _extract_bbox(geo, warnings)
    t_min, t_med = _extract_thickness(geo, bbox, warnings)
    curvature = _extract_curvature(geo)
    orientation = _dominant_orientation(geo, bbox)

    ar   = bbox.aspect_ratio()       # elongation
    flat = bbox.flatness_ratio()     # plate-ness
    curv = curvature * bbox.span     # dimensionless curvature ratio

    kind = _choose_archetype(ar, flat, curv, orientation, load_estimate, warnings)

    cs = _build_cross_section(kind, bbox, t_min, t_med, orientation, warnings)
    eff_len = _effective_length(kind, bbox, orientation)
    sup     = _support_condition(kind, geo, load_estimate)
    if (
        kind == ArchetypeKind.CANTILEVER_BEAM
        and orientation == "horizontal"
        and sup in {"simply_supported", "fixed_fixed"}
    ):
        kind = ArchetypeKind.SIMPLY_SUPPORTED_BEAM

    logger.debug(
        "Classified as %s | ar=%.1f flat=%.3f curv=%.4f eff_len=%.1f mm",
        kind.value, ar, flat, curv, eff_len,
    )

    return StructuralPrimitive(
        kind=kind,
        bbox=bbox,
        cross_section=cs,
        effective_length_mm=eff_len,
        support_condition=sup,
        metadata=_primitive_metadata(geo, bbox, orientation),
    )


# ── Archetype decision tree ────────────────────────────────────────────────────

def _choose_archetype(
    ar: float,
    flat: float,
    curv: float,
    orientation: str,
    load_estimate: dict,
    warnings: list[str],
) -> ArchetypeKind:
    """
    Pure function mapping shape descriptors → ArchetypeKind.

    Decision tree (evaluated top-to-bottom; first match wins):

    1. Shell:      curvature_ratio > threshold
    2. Plate:      very flat (flatness < threshold) AND low elongation
    3. Column:     very elongated AND vertical AND compressive load
    4. Cantilever: elongated AND horizontal AND one-end-free heuristic
    5. SS beam:    elongated AND horizontal AND both-ends-constrained
    6. Block:      compact (low AR, low flatness)
    7. Fallback:   cantilever (conservative)
    """
    stress_mode = load_estimate.get("primary_stress_mode", "bending")

    # ── 1. Shell ─────────────────────────────────────────────────────────
    if curv > _CURVATURE_SHELL and flat < _FLATNESS_PLATE:
        return ArchetypeKind.THIN_WALLED_SHELL

    # ── 2. Plate ─────────────────────────────────────────────────────────
    if flat < _FLATNESS_PLATE:
        if ar > _ASPECT_ELONGATED:
            # Long flat thing — cantilever plate (shelf, wing, flap)
            return ArchetypeKind.CANTILEVER_PLATE
        else:
            # Wide flat thing — flat plate (seat, floor panel, lid)
            return ArchetypeKind.FLAT_PLATE

    # ── 3. Column ────────────────────────────────────────────────────────
    # Only classify as column when the part is VERTICAL (Z-dominant) AND
    # the load mode is compressive. A horizontal elongated member under
    # bending is a beam, not a column.
    if ar > _ASPECT_ELONGATED and orientation == "vertical":
        if stress_mode in ("compression", "none"):
            return ArchetypeKind.SLENDER_COLUMN
        # Vertical + very elongated under non-compressive load:
        # still governs as column (eccentric loading / combined stress)
        if ar > _ASPECT_VERY_ELONGATED:
            return ArchetypeKind.SLENDER_COLUMN
        # Vertical + moderately elongated + bending → cantilever beam
        # (e.g. a mast or post loaded laterally)
        return ArchetypeKind.CANTILEVER_BEAM

    # ── 4 & 5. Beam ──────────────────────────────────────────────────────
    if ar > _ASPECT_ELONGATED:
        # Distinguish cantilever vs simply-supported via support heuristic
        # (caller provides this via load_estimate context or defaults)
        if stress_mode == "bending":
            # Default: cantilever is more conservative; prefer it unless
            # load_type is static + symmetric (implies SS)
            if load_estimate.get("load_type") == "static" and orientation == "horizontal":
                return ArchetypeKind.SIMPLY_SUPPORTED_BEAM
            return ArchetypeKind.CANTILEVER_BEAM
        return ArchetypeKind.CANTILEVER_BEAM

    # ── 6. Solid block ───────────────────────────────────────────────────
    if ar < 3.0 and flat > _FLATNESS_PLATE:
        return ArchetypeKind.SOLID_BLOCK

    # ── 7. Fallback ──────────────────────────────────────────────────────
    warnings.append(
        f"Archetype ambiguous (ar={ar:.1f}, flat={flat:.3f}, curv={curv:.4f}). "
        "Defaulting to cantilever_beam (conservative)."
    )
    return ArchetypeKind.CANTILEVER_BEAM


# ── Cross-section builder ──────────────────────────────────────────────────────

def _build_cross_section(
    kind: ArchetypeKind,
    bbox: BoundingBox,
    t_min: float,
    t_med: float,
    orientation: str,
    warnings: list[str],
) -> CrossSection:
    """
    Choose the structurally appropriate cross-section for each archetype.

    For strength (worst-case): use t_min.
    For stiffness (bulk section): use t_med.
    We store t_min as thickness (conservative for both strength and buckling)
    and let the mechanics functions use t_med where appropriate.
    """
    dims = bbox.sorted_dims   # (min, mid, max)
    d0, d1, d2 = dims         # small, medium, large

    if kind == ArchetypeKind.SLENDER_COLUMN:
        # Column cross-section: perpendicular to long axis
        # Width = mid dim, thickness = min dim (thinnest wall)
        width = d1
        thickness = t_min
        area = width * thickness

    elif kind in (ArchetypeKind.CANTILEVER_BEAM, ArchetypeKind.SIMPLY_SUPPORTED_BEAM):
        # Beam cross-section: perpendicular to span
        # Span = largest dim; section is mid × min
        width = d1
        thickness = t_min
        area = width * thickness

    elif kind in (ArchetypeKind.FLAT_PLATE, ArchetypeKind.CANTILEVER_PLATE):
        # Plate: thickness = smallest dim (the thin direction)
        # Width = mid dim (for per-unit-width we use 1 mm strip → b=1)
        # Store the actual plate dims; formulas will handle the geometry
        width = d1        # plate width
        thickness = t_min if t_min < d0 * 1.5 else d0  # prefer measured
        area = d1 * d2    # plan area

    elif kind == ArchetypeKind.THIN_WALLED_SHELL:
        # Shell: thickness = measured wall; width = circumference proxy
        radius = d1 / 2.0
        width = 2 * math.pi * radius   # circumference
        thickness = t_min
        area = width * thickness

    elif kind == ArchetypeKind.SOLID_BLOCK:
        # Block: full cross-section area
        width = d1
        thickness = d0
        area = d0 * d1

    else:
        width = d1
        thickness = t_min
        area = width * thickness

    # Guard
    MIN_T = 0.01
    if thickness < MIN_T:
        warnings.append(
            f"Cross-section thickness resolved to {thickness:.4f} mm "
            f"(< {MIN_T} mm). Clamping to {MIN_T} mm."
        )
        thickness = MIN_T

    if area < 1e-6:
        area = width * thickness

    return CrossSection(area_mm2=area, width_mm=width, thickness_mm=thickness)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_bbox(geo: dict, warnings: list[str]) -> BoundingBox:
    x = float(geo.get("bbox_x_mm", 0))
    y = float(geo.get("bbox_y_mm", 0))
    z = float(geo.get("bbox_z_mm", 0))
    if x <= 0 or y <= 0 or z <= 0:
        warnings.append(
            f"Non-positive bbox dimension(s): ({x}, {y}, {z}). "
            "Replacing zeros with 1 mm."
        )
        x = max(x, 1.0)
        y = max(y, 1.0)
        z = max(z, 1.0)
    return BoundingBox(x_mm=x, y_mm=y, z_mm=z)


def _extract_thickness(
    geo: dict,
    bbox: BoundingBox,
    warnings: list[str],
) -> tuple[float, float]:
    t_min = geo.get("min_wall_thickness_mm")
    t_med = geo.get("median_wall_thickness_mm")

    thin_dim = bbox.min_dim

    if t_min is None or (isinstance(t_min, float) and t_min <= 0):
        fallback = max(thin_dim / 10.0, 0.1)
        warnings.append(
            f"min_wall_thickness_mm unavailable. "
            f"Using bbox fallback: {fallback:.2f} mm."
        )
        t_min = fallback

    if t_med is None or (isinstance(t_med, float) and t_med <= 0):
        fallback = max(thin_dim / 4.0, 0.1)
        warnings.append(
            f"median_wall_thickness_mm unavailable. "
            f"Using bbox fallback: {fallback:.2f} mm."
        )
        t_med = fallback

    t_min = float(t_min)
    t_med = float(t_med)

    if t_med < t_min:
        t_med = t_min

    if t_med > 0 and t_min < t_med * 0.2:
        robust_floor = min(t_med, thin_dim) * 0.5
        if robust_floor > t_min:
            warnings.append(
                f"min_wall_thickness_mm={t_min:.3f} mm is far below "
                f"median_wall_thickness_mm={t_med:.3f} mm. Treating it as "
                f"a mesh sampling outlier and using {robust_floor:.3f} mm "
                "for structural cross-section sizing."
            )
            t_min = robust_floor

    return t_min, t_med


def _extract_curvature(geo: dict) -> float:
    """Return mean absolute curvature in 1/mm. Default 0 if not available."""
    return float(geo.get("mean_curvature", 0.0))


def _dominant_orientation(geo: dict, bbox: BoundingBox) -> str:
    """
    Determine whether the longest axis is vertical (Z) or horizontal (XY).
    Returns 'vertical' or 'horizontal'.
    
    Uses explicit dominant_axis field first, falls back to bbox comparison.
    """
    dominant_axis = geo.get("dominant_axis", "")

    if dominant_axis == "elongated":
        # Elongated along which axis? Check if Z is the longest dim.
        if bbox.z_mm >= bbox.x_mm and bbox.z_mm >= bbox.y_mm:
            return "vertical"
        return "horizontal"

    if dominant_axis == "flat":
        return "horizontal"   # flat things lie in XY plane

    # Compact / degenerate / unknown → check Z vs max horizontal
    max_horiz = max(bbox.x_mm, bbox.y_mm)
    if bbox.z_mm > max_horiz * 1.3:
        return "vertical"
    return "horizontal"


def _effective_length(
    kind: ArchetypeKind,
    bbox: BoundingBox,
    orientation: str,
) -> float:
    """
    Structural span used in closed-form formulas.

    For beams/columns: the long dimension.
    For plates: the short in-plane dimension (governs bending).
    For shells: the axial length.
    """
    dims = bbox.sorted_dims   # (min, mid, max)

    if kind == ArchetypeKind.SLENDER_COLUMN:
        return bbox.span   # full height

    if kind in (ArchetypeKind.CANTILEVER_BEAM, ArchetypeKind.SIMPLY_SUPPORTED_BEAM):
        return bbox.span   # full beam length

    if kind == ArchetypeKind.FLAT_PLATE:
        # Governing dimension for plate bending: shorter in-plane span
        # (simply-supported: σ and δ governed by shorter span)
        return dims[1]   # mid dim (min is the thickness, max is the longer span)

    if kind == ArchetypeKind.CANTILEVER_PLATE:
        return bbox.span   # cantilevered along the long dim

    if kind == ArchetypeKind.THIN_WALLED_SHELL:
        # Axial length governs meridional stress and column-like buckling
        return bbox.span

    if kind == ArchetypeKind.SOLID_BLOCK:
        return dims[1]   # mid dim (compression height)

    return bbox.span


def _support_condition(
    kind: ArchetypeKind,
    geo: dict,
    load_estimate: dict,
) -> str:
    """
    Infer boundary conditions from geometry metadata.
    Returns one of: 'cantilever', 'simply_supported', 'fixed_fixed'.
    """
    # Mounting holes → fixed at both ends
    if geo.get("has_cylindrical_bores", False):
        return "fixed_fixed"

    # Large bottom contact area relative to footprint → base-supported → SS
    bbox_area = float(geo.get("bbox_x_mm", 1)) * float(geo.get("bbox_y_mm", 1))
    contact_area = float(geo.get("bottom_contact_area_mm2", 0))
    if bbox_area > 0 and contact_area / bbox_area > 0.5:
        return "simply_supported"

    # Plates are typically simply-supported; beams default to cantilever (conservative)
    if kind == ArchetypeKind.FLAT_PLATE:
        return "simply_supported"

    return "cantilever"


def _primitive_metadata(geo: dict, bbox: BoundingBox, orientation: str) -> dict:
    footprint_area = max(bbox.x_mm * bbox.y_mm, 1.0)
    top_contact = max(float(geo.get("top_load_area_mm2", 0.0) or 0.0), 0.0)
    bottom_contact = max(float(geo.get("bottom_contact_area_mm2", 0.0) or 0.0), 0.0)
    return {
        "orientation": orientation,
        "top_contact_area_mm2": top_contact,
        "bottom_contact_area_mm2": bottom_contact,
        "top_contact_ratio": min(top_contact / footprint_area, 1.0),
        "bottom_contact_ratio": min(bottom_contact / footprint_area, 1.0),
        "footprint_area_mm2": footprint_area,
    }
