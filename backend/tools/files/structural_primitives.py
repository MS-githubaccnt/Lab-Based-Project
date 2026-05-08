"""
structural_primitives.py
========================
Dataclasses and enums representing detected structural archetypes.

Each primitive captures the geometry of one mechanically distinct region
of the mesh and carries the closed-form results computed for it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ArchetypeKind(str, Enum):
    SLENDER_COLUMN          = "slender_column"
    CANTILEVER_BEAM         = "cantilever_beam"
    SIMPLY_SUPPORTED_BEAM   = "simply_supported_beam"
    FLAT_PLATE              = "flat_plate"
    CANTILEVER_PLATE        = "cantilever_plate"
    THIN_WALLED_SHELL       = "thin_walled_shell"
    SOLID_BLOCK             = "solid_block"
    UNKNOWN                 = "unknown"


@dataclass
class BoundingBox:
    x_mm: float
    y_mm: float
    z_mm: float

    @property
    def dims(self) -> tuple[float, float, float]:
        return (self.x_mm, self.y_mm, self.z_mm)

    @property
    def sorted_dims(self) -> tuple[float, float, float]:
        """Returns (min, mid, max)."""
        return tuple(sorted(self.dims))  # type: ignore[return-value]

    @property
    def span(self) -> float:
        return max(self.dims)

    @property
    def min_dim(self) -> float:
        return min(self.dims)

    @property
    def mid_dim(self) -> float:
        return self.sorted_dims[1]

    @property
    def volume(self) -> float:
        return self.x_mm * self.y_mm * self.z_mm

    def aspect_ratio(self) -> float:
        """max / min — how elongated is this box."""
        if self.min_dim < 1e-9:
            return float("inf")
        return self.span / self.min_dim

    def flatness_ratio(self) -> float:
        """min / mid — how flat/plate-like. 1.0 = cube-ish, ~0 = very flat."""
        if self.sorted_dims[1] < 1e-9:
            return 0.0
        return self.sorted_dims[0] / self.sorted_dims[1]


@dataclass
class CrossSection:
    """Representative cross-section perpendicular to the load axis."""
    area_mm2: float
    width_mm: float       # larger in-plane dimension
    thickness_mm: float   # smaller in-plane dimension (wall thickness)

    @property
    def I_mm4(self) -> float:
        """Second moment of area about the weaker bending axis (b·t³/12)."""
        return (self.width_mm * self.thickness_mm ** 3) / 12.0

    @property
    def Z_mm3(self) -> float:
        """Section modulus = I / (t/2)."""
        if self.thickness_mm < 1e-9:
            return 0.0
        return self.I_mm4 / (self.thickness_mm / 2.0)

    @property
    def I_weak_mm4(self) -> float:
        """Second moment about weaker axis (t < b → I = b·t³/12 already)."""
        return self.I_mm4


@dataclass
class StructuralPrimitive:
    """One mechanically distinct region of the mesh with its analysis results."""
    kind: ArchetypeKind
    bbox: BoundingBox
    cross_section: CrossSection

    # Filled by load assignment
    applied_force_N: float = 0.0
    applied_pressure_Pa: float = 0.0   # for distributed loads on plates
    effective_length_mm: float = 0.0   # span used in formulas
    support_condition: str = "cantilever"  # cantilever | simply_supported | fixed_fixed

    # Results (filled by mechanics solver)
    stress_MPa: float = 0.0
    required_strength_MPa: float = 0.0
    required_E_GPa: float = 0.0
    buckling_risk: bool = False
    critical_buckling_load_N: float = 0.0
    deflection_mm: float = 0.0
    deflection_limit_mm: float = 0.0

    # Traceability
    metadata: dict = field(default_factory=dict)
    governing_formula: str = ""
    derivation: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "archetype": self.kind.value,
            "effective_length_mm": round(self.effective_length_mm, 2),
            "cross_section_area_mm2": round(self.cross_section.area_mm2, 3),
            "wall_thickness_mm": round(self.cross_section.thickness_mm, 3),
            "applied_force_N": round(self.applied_force_N, 2),
            "applied_pressure_Pa": round(self.applied_pressure_Pa, 2),
            "support_condition": self.support_condition,
            "orientation": self.metadata.get("orientation"),
            "top_contact_ratio": round(float(self.metadata.get("top_contact_ratio", 0.0)), 3),
            "bottom_contact_ratio": round(float(self.metadata.get("bottom_contact_ratio", 0.0)), 3),
            "stress_MPa": round(self.stress_MPa, 2),
            "required_strength_MPa": round(self.required_strength_MPa, 2),
            "required_E_GPa": round(self.required_E_GPa, 3),
            "buckling_risk": self.buckling_risk,
            "deflection_mm": round(self.deflection_mm, 4),
            "deflection_limit_mm": round(self.deflection_limit_mm, 4),
            "governing_formula": self.governing_formula,
            "derivation": self.derivation,
            "warnings": self.warnings,
        }
