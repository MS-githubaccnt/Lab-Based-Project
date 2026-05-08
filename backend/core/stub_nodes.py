"""
nodes/stub_nodes.py
--------------------
Wire-compatible stubs for MLPredictorNode and ReportAssemblyNode.
Each emits thoughts so the frontend
receives reasoning steps even from stub implementations.

Replace each stub with the real implementation when ready.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field
from schema.state import Thought
from agents.supervisor.schemas import PipelineStage
from llm.llm_client import GroqLLMClient


def _t(node: str, type_: str, text: str) -> Thought:
    return Thought(node=node, type=type_, text=text)


class MaterialDescription(BaseModel):
    material_name: str = Field(description="The exact material name from the provided candidates.")
    description: str = Field(
        description=(
            "A short, engineer-facing description of why this material is relevant "
            "for the part. Must be one or two concise sentences."
        )
    )


class MaterialDescriptionResponse(BaseModel):
    descriptions: List[MaterialDescription] = Field(
        description="One description for each supplied material candidate."
    )


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

        # Extract translated requirement features for the material selector.
        ml_input = state.get("ml_input_vector") or {}
        feature_specs = [
            ("required_stiffness_GPa", "Required stiffness", ml_input.get("required_stiffness_GPa", 0.0)),
            (
                "required_tensile_strength_MPa",
                "Required tensile strength",
                ml_input.get("required_tensile_strength_MPa", 0.0),
            ),
        ]
        model_feature_names = [spec[0] for spec in feature_specs]
        feature_labels = {spec[0]: spec[1] for spec in feature_specs}
        feature_values = [float(spec[2] or 0.0) for spec in feature_specs]

        import numpy as np
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Build input array
            input_data = np.array([feature_values])
            
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

        class_labels = list(getattr(self.le, "classes_", []))
        if "probas" in locals() and class_labels:
            ranked = sorted(
                enumerate(zip(class_labels, probas)),
                key=lambda item: float(item[1][1]),
                reverse=True,
            )[:3]
        else:
            pred_index = _safe_class_index(pred_encoded[0] if "pred_encoded" in locals() else None)
            ranked = [(pred_index, (pred_material, confidence))]

        predictions = []
        for class_index, (material_name, score) in ranked:
            score = float(score)
            predictions.append({
                "material_name":           str(material_name),
                "eco_score":               score,
                "confidence":              score,
                "predicted_strength_MPa":  ml_input.get("required_tensile_strength_MPa"),
                "predicted_stiffness_GPa": ml_input.get("required_stiffness_GPa"),
                "description":             "",
                "feature_contributions":   _build_shap_feature_contributions(
                    model=self.pipeline,
                    input_data=input_data,
                    class_index=class_index,
                    feature_names=model_feature_names,
                    feature_labels=feature_labels,
                ),
            })

        descriptions = await _generate_material_descriptions(
            object_description=state.get("object_description", "the uploaded part"),
            load_estimate=state.get("load_estimate") or {},
            ml_input=ml_input,
            predictions=predictions,
        )
        for prediction in predictions:
            material_name = prediction.get("material_name", "")
            prediction["description"] = descriptions.get(
                material_name,
                _fallback_material_description(prediction),
            )

        for i, p in enumerate(predictions, 1):
            new_thoughts.append(_t(node, "result",
                f"#{i}: {p['material_name']} "
                f"(score {p['confidence']:.2f})"
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


async def _generate_material_descriptions(
    object_description: str,
    load_estimate: Dict[str, Any],
    ml_input: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Dict[str, str]:
    if not predictions:
        return {}

    fallback = {
        prediction.get("material_name", ""): _fallback_material_description(prediction)
        for prediction in predictions
    }

    try:
        llm = GroqLLMClient(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            system_prompt=(
                "You write concise engineering material descriptions. "
                "Do not invent exact properties that were not provided. "
                "Return practical, non-marketing language."
            ),
        )
        response = await llm.generate(
            messages=[{
                "role": "user",
                "content": _build_description_prompt(
                    object_description=object_description,
                    load_estimate=load_estimate,
                    ml_input=ml_input,
                    predictions=predictions,
                ),
            }],
            response_model=MaterialDescriptionResponse,
        )
    except Exception:
        return fallback

    descriptions = dict(fallback)
    material_names_by_key = {
        str(prediction.get("material_name", "")).strip().lower(): str(prediction.get("material_name", "")).strip()
        for prediction in predictions
    }
    for item in response.descriptions:
        name = item.material_name.strip()
        description = item.description.strip()
        if name and description:
            canonical_name = material_names_by_key.get(name.lower(), name)
            descriptions[canonical_name] = description

    return descriptions


def _build_description_prompt(
    object_description: str,
    load_estimate: Dict[str, Any],
    ml_input: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> str:
    candidate_lines = []
    for index, prediction in enumerate(predictions[:3], 1):
        candidate_lines.append(
            "- "
            f"rank {index}: {prediction.get('material_name', 'Unknown')} | "
            f"score={prediction.get('confidence', 'unknown')} | "
            f"required_strength_MPa={prediction.get('predicted_strength_MPa', 'unknown')} | "
            f"required_stiffness_GPa={prediction.get('predicted_stiffness_GPa', 'unknown')}"
        )

    return (
        "Generate a short dropdown description for each of the top material candidates.\n"
        "Keep each description to one or two sentences. Explain why the material is a relevant candidate "
        "for the part and mention any tradeoff only if it follows from the provided context.\n\n"
        f"Part: {object_description}\n"
        f"Load type: {load_estimate.get('load_type', 'unknown')}\n"
        f"Primary stress mode: {load_estimate.get('primary_stress_mode', 'unknown')}\n"
        f"Expected safety factor: {load_estimate.get('safety_factor', 'unknown')}\n"
        f"Required tensile strength MPa: {ml_input.get('required_tensile_strength_MPa', 'unknown')}\n"
        f"Required stiffness GPa: {ml_input.get('required_stiffness_GPa', 'unknown')}\n"
        f"Buckling risk: {ml_input.get('buckling_risk', 'unknown')}\n"
        f"Fatigue critical: {ml_input.get('fatigue_critical', 'unknown')}\n\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
    )


def _fallback_material_description(prediction: Dict[str, Any]) -> str:
    material_name = prediction.get("material_name", "This material")
    score = prediction.get("confidence")
    score_text = f" with a model score of {score:.2f}" if isinstance(score, float) else ""
    return (
        f"{material_name} is a candidate match{score_text} for the translated load "
        "and geometry requirements. Review detailed material datasheets before final selection."
    )


def _safe_class_index(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _build_shap_feature_contributions(
    model: Any,
    input_data: Any,
    class_index: int,
    feature_names: List[str],
    feature_labels: Dict[str, str],
) -> List[Dict[str, Any]]:
    try:
        import numpy as np
        import shap

        background = np.array([
            [0.5, 10.0],
            [1.5, 25.0],
            [3.0, 50.0],
            [8.0, 100.0],
            [20.0, 200.0],
        ])
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(input_data, nsamples=32)
        class_values = _extract_class_shap_values(shap_values, class_index)

        contributions = []
        for feature_name, value, shap_value in zip(feature_names, input_data[0], class_values):
            shap_value = float(shap_value)
            contributions.append({
                "feature": feature_name,
                "label": feature_labels.get(feature_name, feature_name),
                "value": float(value),
                "shap_value": shap_value,
                "impact": abs(shap_value),
                "direction": "supports" if shap_value >= 0 else "opposes",
            })

        return sorted(contributions, key=lambda item: item["impact"], reverse=True)
    except Exception:
        return _fallback_feature_contributions(input_data, feature_names, feature_labels)


def _extract_class_shap_values(shap_values: Any, class_index: int) -> List[float]:
    if isinstance(shap_values, list):
        index = min(max(class_index, 0), len(shap_values) - 1)
        return list(shap_values[index][0])

    shape = getattr(shap_values, "shape", ())
    if len(shape) == 3:
        index = min(max(class_index, 0), shape[2] - 1)
        return list(shap_values[0, :, index])
    if len(shape) == 2:
        return list(shap_values[0])

    return [0.0, 0.0]


def _fallback_feature_contributions(
    input_data: Any,
    feature_names: List[str],
    feature_labels: Dict[str, str],
) -> List[Dict[str, Any]]:
    values = list(input_data[0])
    total = sum(abs(float(value)) for value in values) or 1.0

    contributions = []
    for feature_name, value in zip(feature_names, values):
        value = float(value)
        impact = abs(value) / total
        contributions.append({
            "feature": feature_name,
            "label": feature_labels.get(feature_name, feature_name),
            "value": value,
            "shap_value": 0.0,
            "impact": impact,
            "direction": "supports",
        })

    return sorted(contributions, key=lambda item: item["impact"], reverse=True)
