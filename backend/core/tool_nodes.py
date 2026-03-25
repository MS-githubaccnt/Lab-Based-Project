"""
nodes/tool_nodes.py
--------------------
Thin LangGraph node wrappers around the two deterministic tools.
Each node emits thoughts: List[Thought] describing what it computed,
so the frontend can show real-time reasoning steps.
"""

from __future__ import annotations

from typing import Any, Dict, List

from tools.cad_parser_tools import parse_and_extract_cad_features
from tools.feature_translation_tools import translate_features_to_ml_inputs
from schema.state import Thought
from agents.supervisor.schemas import PipelineStage


def _t(node: str, type_: str, text: str) -> Thought:
    return Thought(node=node, type=type_, text=text)


class CadParserNode:
    """
    Parses the CAD file and extracts geometry features.
    Reads:  cad_file_path
    Writes: geometry, thoughts (appended), pipeline_stage
    """

    def __init__(
        self,
        tessellation_tolerance_mm: float = 0.1,
        wall_sample_count: int = 300,
        thin_wall_threshold_mm: float = 1.5,
    ) -> None:
        self.tessellation_tolerance_mm = tessellation_tolerance_mm
        self.wall_sample_count         = wall_sample_count
        self.thin_wall_threshold_mm    = thin_wall_threshold_mm

    async def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        file_path = state.get("cad_file_path")
        if not file_path:
            raise KeyError(
                "CadParserNode requires state['cad_file_path']. "
                "Set it in the initial state passed to process_request()."
            )

        existing = list(state.get("thoughts") or [])
        node     = "cad_parser"
        new_thoughts: List[Thought] = [
            _t(node, "info", f"Parsing CAD file: {file_path.split('/')[-1]}")
        ]

        geometry = parse_and_extract_cad_features.invoke({
            "file_path":                   file_path,
            "tessellation_tolerance_mm":   self.tessellation_tolerance_mm,
            "wall_sample_count":           self.wall_sample_count,
            "thin_wall_threshold_mm":      self.thin_wall_threshold_mm,
        })

        # Emit computed geometry facts as result thoughts
        fmt = geometry.get("source_format", "unknown")
        tri = geometry.get("triangle_count", "?")
        new_thoughts.append(_t(node, "result",
            f"Loaded {tri:,} triangles ({fmt})"
            if isinstance(tri, int) else f"Loaded mesh ({fmt})"
        ))

        axis = geometry.get("dominant_axis", "?")
        ar   = geometry.get("aspect_ratio")
        ar_s = f", aspect ratio {ar:.1f}" if ar is not None else ""
        new_thoughts.append(_t(node, "result",
            f"Shape: {axis}{ar_s}"
        ))

        t_min = geometry.get("min_wall_thickness_mm")
        t_med = geometry.get("median_wall_thickness_mm")
        vol   = geometry.get("volume_mm3")
        wall_s = (
            f"Min wall: {t_min:.2f} mm | Median wall: {t_med:.2f} mm"
            if t_min is not None and t_med is not None
            else "Wall thickness: not measured (open mesh)"
        )
        vol_s  = f" | Volume: {vol:,.0f} mm³" if vol is not None else ""
        new_thoughts.append(_t(node, "result", wall_s + vol_s))

        if not geometry.get("is_watertight", True):
            new_thoughts.append(_t(node, "warning",
                "Mesh is not watertight — wall thickness estimates are approximate"
            ))

        for w in (geometry.get("warnings") or []):
            new_thoughts.append(_t(node, "warning", w))

        return {
            "geometry":       geometry,
            "thoughts":       existing + new_thoughts,
            "pipeline_stage": PipelineStage.CAD_PARSER,
        }


class FeatureTranslationNode:
    """
    Derives the ML input vector from geometry + load estimate.
    Reads:  geometry, load_estimate, recyclability_priority
    Writes: ml_input_vector, thoughts (appended), pipeline_stage
    """

    async def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        geometry = state.get("geometry")
        if not geometry:
            raise KeyError(
                "FeatureTranslationNode requires state['geometry']. "
                "Ensure CadParserNode ran successfully."
            )
        load_estimate = state.get("load_estimate")
        if not load_estimate:
            raise KeyError(
                "FeatureTranslationNode requires state['load_estimate']. "
                "Ensure LoadInferenceNode ran successfully."
            )

        existing = list(state.get("thoughts") or [])
        node     = "feature_translation"
        new_thoughts: List[Thought] = [
            _t(node, "info", "Deriving mechanical requirements from geometry + load...")
        ]

        ml_input_vector = translate_features_to_ml_inputs.invoke({
            "geometry":               geometry,
            "load_estimate":          load_estimate,
            "recyclability_priority": state.get("recyclability_priority", 0.7),
        })

        strength  = ml_input_vector.get("required_tensile_strength_MPa")
        stiffness = ml_input_vector.get("required_stiffness_GPa")
        dominated = ml_input_vector.get("stiffness_dominated", False)
        buckling  = ml_input_vector.get("buckling_risk", False)

        if strength is not None:
            new_thoughts.append(_t(node, "result",
                f"Required tensile strength: {strength:,.0f} MPa"
            ))
        if stiffness is not None:
            suffix = " [geometry-dominated — no material fully meets this]" if dominated else ""
            new_thoughts.append(_t(node, "result",
                f"Required stiffness: {stiffness:.1f} GPa{suffix}"
            ))
        if buckling:
            new_thoughts.append(_t(node, "warning",
                "Euler buckling risk flagged — consider adding lateral support"
            ))
        for w in (ml_input_vector.get("warnings") or []):
            new_thoughts.append(_t(node, "warning", w))

        return {
            "ml_input_vector": ml_input_vector,
            "thoughts":        existing + new_thoughts,
            "pipeline_stage":  PipelineStage.FEATURE_TRANSLATION,
        }