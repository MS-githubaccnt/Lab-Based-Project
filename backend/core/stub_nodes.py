"""
nodes/stub_nodes.py
--------------------
Wire-compatible stubs for ClarificationNode, MLPredictorNode,
and ReportAssemblyNode. Each emits thoughts so the frontend
receives reasoning steps even from stub implementations.

Replace each stub with the real implementation when ready.
"""

from __future__ import annotations

from typing import Any, Dict, List

from schema.state import Thought
from agents.supervisor.schemas import PipelineStage


def _t(node: str, type_: str, text: str) -> Thought:
    return Thought(node=node, type=type_, text=text)


class ClarificationNode:
    """
    Human-in-loop pause. The real implementation relies on
    interrupt_before=['clarification'] in the compiled graph.
    By the time invoke() runs, clarification_answer is already in state.
    """

    async def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        existing = list(state.get("thoughts") or [])
        return {
            "thoughts":       existing + [_t("clarification", "info",
                "User clarification received — refining load estimate...")],
            "pipeline_stage": PipelineStage.CLARIFICATION,
        }


class MLPredictorNode:
    """
    Calls the trained eco-material ML model.
    """

    def __init__(self):
        import joblib
        import warnings
        from pathlib import Path
        
        base_dir = Path(__file__).resolve().parent
        
        # Suppress scikit-learn unpickling warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipeline = joblib.load(base_dir / "material_classifier.pkl")
            self.le = joblib.load(base_dir / "label_encoder.pkl")

    async def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        existing = list(state.get("thoughts") or [])
        node     = "ml_predictor"
        new_thoughts: List[Thought] = [
            _t(node, "info", "Running eco-material selection model...")
        ]

        # Extract features from state
        ml_input = state.get("ml_input_vector") or {}
        modulus = ml_input.get("tensile_modulus_GPa", 0.0)
        strength = ml_input.get("tensile_strength_MPa", 0.0)

        import numpy as np
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Build input array
            input_data = np.array([[modulus, strength]])  # modulus, strength
            
            try:
                # Perform prediction
                pred_encoded = self.pipeline.predict(input_data)
                pred_material = self.le.inverse_transform(pred_encoded)[0]
                
                # Try getting prediction confidence
                if hasattr(self.pipeline, "predict_proba"):
                    probas = self.pipeline.predict_proba(input_data)[0]
                    confidence = float(np.max(probas))
                else:
                    confidence = 0.90
            except Exception as e:
                pred_material = "Unknown"
                confidence = 0.0
                new_thoughts.append(_t(node, "warning", f"Model prediction error: {e}"))

        # Model only provides material_name based on input snippet
        predictions = [{
            "material_name":           pred_material,
            "eco_score":               0.80, # Stub eco score
            "confidence":              confidence,
            "predicted_strength_MPa":  None,
            "predicted_stiffness_GPa": None,
        }]

        for i, p in enumerate(predictions, 1):
            new_thoughts.append(_t(node, "result",
                f"#{i}: {p['material_name']} "
                f"(eco {p['eco_score']:.2f} | conf {p['confidence']:.2f})"
            ))

        return {
            "predictions":    predictions,
            "thoughts":       existing + new_thoughts,
            "pipeline_stage": PipelineStage.ML_PREDICTOR,
        }


class ReportAssemblyNode:
    """
    Assembles the FinalReport dict from all upstream state fields.
    No LLM. Pure data collection.

    TODO: Expand the report schema to match your frontend's expected shape.
    """

    async def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        existing    = list(state.get("thoughts") or [])
        node        = "report_assembly"
        predictions = state.get("predictions") or []
        explanation = state.get("explanation") or {}
        warnings    = state.get("supervisor_warnings") or []

        new_thoughts: List[Thought] = [
            _t(node, "info", "Assembling final report...")
        ]

        report = {
            "top_recommendation": predictions[0] if predictions else None,
            "alternatives":       predictions[1:3] if len(predictions) > 1 else [],
            "explanation":        explanation,
            "load_estimate":      state.get("load_estimate"),
            "mechanical_requirements": state.get("ml_input_vector"),
            "geometry_summary": {
                k: (state.get("geometry") or {}).get(k)
                for k in (
                    "dominant_axis", "aspect_ratio",
                    "min_wall_thickness_mm", "volume_mm3",
                    "is_watertight", "source_format",
                )
            },
            "inference_confidence": state.get("inference_confidence"),
            "clarification_used":   state.get("clarification_answer") is not None,
            "warnings":             warnings,
            "all_thoughts":         existing,  # full reasoning trace in the report
        }

        if report["top_recommendation"]:
            name = report["top_recommendation"].get("material_name", "?")
            eco  = report["top_recommendation"].get("eco_score", "?")
            new_thoughts.append(_t(node, "result",
                f"Report assembled — top material: {name} (eco {eco:.2f})"
                if isinstance(eco, float) else f"Report assembled — top material: {name}"
            ))

        return {
            "report":         report,
            "thoughts":       existing + new_thoughts,
            "pipeline_stage": "report_assembly",
        }