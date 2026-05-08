"""
load_distributor.py
===================
Assigns applied forces to each StructuralPrimitive based on structural
topology and load type.

Design contract
---------------
- No LLM. Fully deterministic.
- Implements statically determinate force distribution.
- Falls back to conservative (worst-case) assumptions when topology
  is ambiguous.

Load distribution strategy
--------------------------
The key insight: force flows from load entry point to support via the
stiffest path. We approximate this with two rules:

Rule 1 — Series path (load flows through primitives in sequence):
    All primitives carry the full load.
    Used when: primitives are stacked vertically (columns under a plate).

Rule 2 — Parallel path (load splits among equal primitives):
    Each primitive carries load / n_primitives.
    Used when: multiple identical primitives side-by-side (leg array).

We detect which rule applies using the spatial relationship between
component bounding boxes (if available) or fall back to Rule 1
(conservative — every part sees full load).

For distributed loads (plates): convert total force to pressure
using the plan area of the receiving surface.
"""
from __future__ import annotations

import math
import logging
from typing import Optional

try:
    from .structural_primitives import ArchetypeKind, StructuralPrimitive
except ImportError:  # pragma: no cover - supports direct script execution in tools/files
    from structural_primitives import ArchetypeKind, StructuralPrimitive

logger = logging.getLogger("feature-translation")

# Force fraction applied horizontally to back/side members (e.g. lateral loads)
_LATERAL_FRACTION = 0.3   # 30% of vertical load as horizontal (wind, lean, push)


def distribute_loads(
    primitives: list[StructuralPrimitive],
    load: dict,
    geometry: dict,
    warnings: list[str],
) -> None:
    """
    Assign applied_force_N and applied_pressure_Pa to each primitive.
    Modifies primitives in-place.

    Parameters
    ----------
    primitives :
        List of StructuralPrimitive from archetype_detector.
    load :
        Parsed load dict from _parse_load_estimate.
        Keys used: mag_max_N, primary_stress_mode, load_type.
    geometry :
        Raw geometry dict (for plan area, contact area etc.).
    warnings :
        Mutable list; warnings appended in-place.
    """
    F_total = load["mag_max_N"]

    if F_total == 0.0:
        for p in primitives:
            p.applied_force_N = 0.0
        return

    # ── Classify primitives by structural role ─────────────────────────────
    columns = [p for p in primitives if p.kind == ArchetypeKind.SLENDER_COLUMN]
    plates  = [p for p in primitives
                if p.kind in (ArchetypeKind.FLAT_PLATE, ArchetypeKind.CANTILEVER_PLATE)]
    beams   = [p for p in primitives
                if p.kind in (ArchetypeKind.CANTILEVER_BEAM,
                              ArchetypeKind.SIMPLY_SUPPORTED_BEAM)]
    shells  = [p for p in primitives if p.kind == ArchetypeKind.THIN_WALLED_SHELL]
    blocks  = [p for p in primitives if p.kind == ArchetypeKind.SOLID_BLOCK]
    others  = [p for p in primitives
                if p not in columns + plates + beams + shells + blocks]

    stress_mode = load.get("primary_stress_mode", "bending")
    load_type   = load.get("load_type", "static")

    if stress_mode == "compression":
        _distribute_compression_load(
            primitives=primitives,
            columns=columns,
            plates=plates,
            beams=beams,
            shells=shells,
            blocks=blocks,
            others=others,
            F_total=F_total,
            warnings=warnings,
        )
        if load_type in ("cyclic", "impact"):
            _augment_lateral(beams + [p for p in primitives
                                      if p.kind == ArchetypeKind.CANTILEVER_PLATE],
                             F_total, warnings)
        _log_distribution(primitives)
        return

    # ── Columns: parallel path — load splits evenly ────────────────────────
    n_cols = len(columns)
    for col in columns:
        # Each column carries an equal share of the axial load
        col.applied_force_N = F_total / n_cols if n_cols > 0 else 0.0

        # Add eccentric component: lateral fraction creates bending moment
        # but we encode it as force on the column (the eccentric solver handles it)
        # Nothing extra needed here — eccentricity is modelled inside the solver.

    # ── Plates: full load distributed as pressure ──────────────────────────
    for plate in plates:
        plate.applied_force_N = F_total   # full load on the plate
        a = plate.effective_length_mm
        b = plate.bbox.span
        plan_area = max(a * b, 1.0)
        plate.applied_pressure_Pa = (F_total / plan_area) * 1e6  # Pa (N/m² = N/mm² × 1e6)

    # ── Beams: series path by default — full load ──────────────────────────
    # If multiple beams exist in parallel (e.g., twin rails), split.
    n_beams = len(beams)
    for beam in beams:
        if stress_mode in ("bending", "torsion"):
            # Horizontal members under transverse load: full load if single,
            # halved if multiple
            beam.applied_force_N = F_total / max(n_beams, 1)
        else:
            beam.applied_force_N = F_total / max(n_beams, 1)

    # ── Shells: full radial load ───────────────────────────────────────────
    for shell in shells:
        shell.applied_force_N = F_total

    # ── Solid blocks: full bearing load ───────────────────────────────────
    n_blocks = len(blocks)
    for block in blocks:
        block.applied_force_N = F_total / max(n_blocks, 1)

    # ── Others / unknowns: conservative full load ──────────────────────────
    for p in others:
        p.applied_force_N = F_total

    # ── Lateral (horizontal) augmentation ─────────────────────────────────
    # Beams and cantilever plates may experience lateral loads in addition
    # to the primary vertical load.  We model this as a fraction of the
    # vertical load applied transversely — giving a more severe stress.
    if load_type in ("cyclic", "impact"):
        _augment_lateral(beams + [p for p in primitives
                                  if p.kind == ArchetypeKind.CANTILEVER_PLATE],
                         F_total, warnings)

    _log_distribution(primitives)


def _distribute_compression_load(
    primitives: list[StructuralPrimitive],
    columns: list[StructuralPrimitive],
    plates: list[StructuralPrimitive],
    beams: list[StructuralPrimitive],
    shells: list[StructuralPrimitive],
    blocks: list[StructuralPrimitive],
    others: list[StructuralPrimitive],
    F_total: float,
    warnings: list[str],
) -> None:
    """
    Route vertical compression through plausible load paths. Horizontal beams
    are not primary vertical load carriers unless contact evidence says they
    directly receive or support the load.
    """
    axial_supports = [
        p for p in columns
        if p.metadata.get("orientation") == "vertical"
        or p.metadata.get("bottom_contact_ratio", 0.0) > 0.01
    ]

    top_receivers = [
        p for p in primitives
        if _is_compression_top_receiver(p)
    ]

    contact_beams = [
        p for p in beams
        if p in top_receivers
        or p.metadata.get("bottom_contact_ratio", 0.0) > 0.15
    ]

    if axial_supports:
        share = F_total / len(axial_supports)
        for support in axial_supports:
            support.applied_force_N = share

        top_area_total = sum(
            max(float(receiver.metadata.get("top_contact_area_mm2", 0.0)), 0.0)
            for receiver in top_receivers
        )
        for receiver in top_receivers:
            top_area = max(float(receiver.metadata.get("top_contact_area_mm2", 0.0)), 0.0)
            receiver_force = (
                F_total * top_area / top_area_total
                if top_area_total > 0
                else F_total / max(len(top_receivers), 1)
            )
            receiver.applied_force_N = max(receiver.applied_force_N, receiver_force)
            if receiver.kind in (ArchetypeKind.FLAT_PLATE, ArchetypeKind.CANTILEVER_PLATE):
                a = receiver.effective_length_mm
                b = receiver.bbox.span
                plan_area = max(a * b, 1.0)
                receiver.applied_pressure_Pa = (receiver_force / plan_area) * 1e6

        skipped = [p for p in beams if p not in contact_beams]
        if skipped:
            warnings.append(
                f"Compression load path: {len(skipped)} horizontal beam-like "
                "component(s) are not on direct top/bottom bearing contact; "
                "excluded from primary vertical load distribution."
            )
        return

    fallback_carriers = top_receivers + contact_beams + blocks + shells + others
    if not fallback_carriers:
        fallback_carriers = primitives

    warnings.append(
        "Compression load path: no clear axial supports detected. "
        "Distributing load across direct-contact/load-bearing components."
    )
    share = F_total / max(len(fallback_carriers), 1)
    for p in fallback_carriers:
        p.applied_force_N = share


def _is_compression_top_receiver(p: StructuralPrimitive) -> bool:
    top_ratio = p.metadata.get("top_contact_ratio", 0.0)
    if p.kind in (ArchetypeKind.CANTILEVER_BEAM, ArchetypeKind.SIMPLY_SUPPORTED_BEAM):
        return top_ratio > 0.15
    return top_ratio > 0.02


def _augment_lateral(
    members: list[StructuralPrimitive],
    F_total: float,
    warnings: list[str],
) -> None:
    """
    For dynamic / impact loads, augment beam/plate loads with a lateral
    component (30% of vertical — ASCE 7-22 lateral live load allowance).
    Takes the larger of existing or lateral force.
    """
    F_lateral = F_total * _LATERAL_FRACTION
    for m in members:
        if m.applied_force_N < F_lateral:
            warnings.append(
                f"Dynamic load augmentation: {m.kind.value} force increased "
                f"from {m.applied_force_N:.0f} N to {F_lateral:.0f} N "
                f"(lateral fraction {_LATERAL_FRACTION*100:.0f}% of total)."
            )
            m.applied_force_N = F_lateral


def _log_distribution(primitives: list[StructuralPrimitive]) -> None:
    for p in primitives:
        logger.debug(
            "  %s → F=%.1f N", p.kind.value, p.applied_force_N
        )
