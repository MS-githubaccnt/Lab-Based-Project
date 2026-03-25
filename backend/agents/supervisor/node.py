"""
supervisor/node.py
-------------------
SupervisorNode: owns all routing decisions and quality gates for the
eco-material selection pipeline.

Architecture
------------
The supervisor sits between every pair of nodes. Each upstream node writes
pipeline_stage = '<its own name>' to state before returning. The supervisor
reads this, runs the corresponding quality gate, and returns a LangGraph
Command that routes to the next node (or END on abort).

LLM usage
---------
The supervisor calls the LLM in exactly ONE case: generating a user-facing
AbortReason when the pipeline must stop. For all routing, warn, and retry
decisions the logic is fully deterministic.

This means the typical pipeline run involves zero LLM calls from the
supervisor. The LLM is there only to produce a helpful error message
when something goes wrong — not to make decisions.

LangGraph integration
---------------------
The supervisor node is compiled with ALL edges running through it:

    builder.add_node("supervisor", supervisor_node)
    builder.set_entry_point("supervisor")

    # Each content node routes back to supervisor on completion
    for node in [cad_parser, load_inference, clarification,
                 feature_translation, ml_predictor, explanation,
                 report_assembly]:
        builder.add_node(node.__name__, node)
        builder.add_edge(node.__name__, "supervisor")

    # Supervisor routes forward via Command(goto=...)
    builder.add_conditional_edges("supervisor", lambda s: s["_next"],
                                   {n: n for n in ALL_NODES + [END]})

State fields consumed
---------------------
    pipeline_stage          str     — which node just completed
    geometry                dict    — from cad_parser
    load_estimate           dict    — from load_inference
    inference_confidence    float   — from load_inference
    ml_input_vector         dict    — from feature_translation
    predictions             list    — from ml_predictor
    load_inference_retries  int     — supervisor-managed retry counter

State fields written
---------------------
    supervisor_warnings     list[str]   — accumulated non-fatal issues
    pipeline_error          dict|None   — AbortReason on abort, else None
    load_inference_retries  int         — incremented on retry
    _next                   str         — next node name for routing
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langgraph.types import Command  # type: ignore

from agents.base_agent import BaseAgentNode
from .prompt_builder import SupervisorPromptBuilder
from .schemas import (
    AbortReason,
    NextNode,
    PipelineStage,
    QualityThresholds as QT,
)


DEFAULT_MODEL       = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.1   # deterministic — only used for abort messages


class SupervisorNode(BaseAgentNode):
    """
    Routes the pipeline, enforces quality gates, and generates
    user-facing error messages on abort.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        thresholds: QT = None,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            system_prompt=SupervisorPromptBuilder.build_main_prompt(),
        )
        # Allow threshold overrides for testing
        self.thresholds = thresholds or QT()

    async def invoke(self, state: Dict[str, Any]) -> Command:
        """
        Inspect state, run the quality gate for the just-completed stage,
        and return a Command routing to the next node.

        Returns:
            langgraph.types.Command with:
              goto   — name of the next node (or END)
              update — state delta (warnings, error, retry counter, _next)
        """
        stage = state.get("pipeline_stage")

        # First invocation: no stage set yet — start the pipeline
        if stage is None:
            return Command(
                goto=NextNode.CAD_PARSER,
                update={"_next": NextNode.CAD_PARSER,
                        "supervisor_warnings": [],
                        "pipeline_error": None,
                        "load_inference_retries": 0},
            )

        # Dispatch to the gate for the stage that just completed
        dispatch = {
            PipelineStage.CAD_PARSER:          self._gate_cad_parser,
            PipelineStage.LOAD_INFERENCE:      self._gate_load_inference,
            PipelineStage.CLARIFICATION:       self._gate_clarification,
            PipelineStage.FEATURE_TRANSLATION: self._gate_feature_translation,
            PipelineStage.ML_PREDICTOR:        self._gate_ml_predictor,
            PipelineStage.EXPLANATION:         self._gate_explanation,
        }

        gate_fn = dispatch.get(stage)
        if gate_fn is None:
            # Unknown stage — abort with a clear message rather than silently
            # routing somewhere unexpected.
            return await self._abort(
                stage=str(stage),
                trigger=f"Unrecognised pipeline_stage '{stage}'.",
                state_snapshot={"pipeline_stage": stage},
                state=state,
            )

        return await gate_fn(state)

    # =======================================================================
    # Quality gates — one per pipeline stage
    # =======================================================================

    async def _gate_cad_parser(self, state: Dict[str, Any]) -> Command:
        """
        Validate geometry output from cad_parser.

        Abort conditions (unrecoverable):
          - dominant_axis == 'degenerate'  (near-zero dimension)
          - volume_mm3 <= 0                (empty / inside-out mesh)
          - triangle_count < MIN          (too sparse for any analysis)

        Warnings (proceed):
          - not is_watertight             (wall thickness less reliable)
          - min_wall_thickness_mm is None (fell back to bbox estimate)
        """
        geo      = state.get("geometry", {})
        warnings = list(state.get("supervisor_warnings", []))

        # ── Abort: degenerate geometry ─────────────────────────────────
        if geo.get("dominant_axis") == "degenerate":
            return await self._abort(
                stage=PipelineStage.CAD_PARSER,
                trigger="dominant_axis == 'degenerate' (near-zero bounding-box dimension)",
                state_snapshot={
                    "bbox_x_mm": geo.get("bbox_x_mm"),
                    "bbox_y_mm": geo.get("bbox_y_mm"),
                    "bbox_z_mm": geo.get("bbox_z_mm"),
                    "source_format": geo.get("source_format"),
                },
                state=state,
            )

        # ── Abort: zero or negative volume ─────────────────────────────
        volume = geo.get("volume_mm3", 0.0) or 0.0
        if volume <= self.thresholds.MIN_VOLUME_MM3:
            return await self._abort(
                stage=PipelineStage.CAD_PARSER,
                trigger=f"volume_mm3 = {volume} <= 0 (empty or inside-out mesh)",
                state_snapshot={
                    "volume_mm3": volume,
                    "is_watertight": geo.get("is_watertight"),
                    "source_format": geo.get("source_format"),
                },
                state=state,
            )

        # ── Abort: too few triangles ────────────────────────────────────
        tri_count = geo.get("triangle_count", 0) or 0
        if tri_count < self.thresholds.MIN_TRIANGLE_COUNT:
            return await self._abort(
                stage=PipelineStage.CAD_PARSER,
                trigger=(
                    f"triangle_count = {tri_count} < {self.thresholds.MIN_TRIANGLE_COUNT} "
                    "(insufficient geometry for analysis)"
                ),
                state_snapshot={
                    "triangle_count": tri_count,
                    "source_format": geo.get("source_format"),
                },
                state=state,
            )

        # ── Warn: open mesh ────────────────────────────────────────────
        if not geo.get("is_watertight", True):
            warnings.append(
                "Mesh is not watertight. Wall thickness estimates and volume "
                "may be less accurate. Results are indicative."
            )

        # ── Warn: wall thickness fell back to bbox estimate ─────────────
        if geo.get("min_wall_thickness_mm") is None:
            warnings.append(
                "Wall thickness could not be measured from ray-casting "
                "(open mesh). A bounding-box estimate was used. "
                "Strength calculation is approximate."
            )

        return Command(
            goto=NextNode.LOAD_INFERENCE,
            update={
                "_next": NextNode.LOAD_INFERENCE,
                "supervisor_warnings": warnings,
            },
        )

    async def _gate_load_inference(self, state: Dict[str, Any]) -> Command:
        """
        Validate load_estimate output from load_inference.

        Routes:
          confidence < threshold        -> clarification
          zero force on mechanical part -> retry (once)
          retries exhausted             -> abort
          all checks pass               -> feature_translation

        Warnings (proceed):
          safety_factor < MIN_SAFETY_FACTOR
        """
        load     = state.get("load_estimate", {})
        warnings = list(state.get("supervisor_warnings", []))
        retries  = state.get("load_inference_retries", 0)
        conf     = state.get("inference_confidence", 1.0)

        # ── Abort: retry budget exhausted ──────────────────────────────
        if retries >= self.thresholds.MAX_LOAD_INFERENCE_RETRIES:
            return await self._abort(
                stage=PipelineStage.LOAD_INFERENCE,
                trigger=(
                    f"load_inference_retries = {retries} >= "
                    f"{self.thresholds.MAX_LOAD_INFERENCE_RETRIES}. "
                    "Pipeline could not produce a confident load estimate "
                    "even after clarification."
                ),
                state_snapshot={
                    "object_description": state.get("object_description"),
                    "inference_confidence": conf,
                    "load_type": load.get("load_type"),
                    "primary_stress_mode": load.get("primary_stress_mode"),
                },
                state=state,
            )

        # ── Route: low confidence -> clarification ─────────────────────
        if conf < self.thresholds.CONFIDENCE_THRESHOLD:
            return Command(
                goto=NextNode.CLARIFICATION,
                update={
                    "_next": NextNode.CLARIFICATION,
                    "supervisor_warnings": warnings,
                },
            )

        # ── Retry: zero magnitude on a mechanical part ─────────────────
        mag_max   = (load.get("magnitude_range_N") or [0, 0])[1]
        load_type = load.get("load_type", "static")
        if mag_max == 0.0 and load_type != "thermal":
            warnings.append(
                f"Load inference returned magnitude_range_N = [0, 0] for a "
                f"'{load_type}' part. Retrying with an explicit prompt that "
                "the part carries non-zero mechanical load."
            )
            return Command(
                goto=NextNode.LOAD_INFERENCE,
                update={
                    "_next": NextNode.LOAD_INFERENCE,
                    "supervisor_warnings": warnings,
                    "load_inference_retries": retries + 1,
                    # Inject a clarification hint so load_inference knows why it's retrying
                    "clarification_answer": (
                        "The part carries a non-zero mechanical load. "
                        "Please estimate a realistic force magnitude range "
                        "for this object class."
                    ),
                },
            )

        # ── Warn: aggressive safety factor ────────────────────────────
        sf = load.get("safety_factor", 2.0)
        if sf < self.thresholds.MIN_SAFETY_FACTOR:
            warnings.append(
                f"Safety factor of {sf:.2f} is below the recommended minimum "
                f"of {self.thresholds.MIN_SAFETY_FACTOR}. This is very aggressive. "
                "The recommendation assumes no additional margin exists elsewhere "
                "in the design."
            )

        return Command(
            goto=NextNode.FEATURE_TRANSLATION,
            update={
                "_next": NextNode.FEATURE_TRANSLATION,
                "supervisor_warnings": warnings,
            },
        )

    async def _gate_clarification(self, state: Dict[str, Any]) -> Command:
        """
        After clarification, always route back to load_inference.
        The clarification_answer is already in state.
        """
        return Command(
            goto=NextNode.LOAD_INFERENCE,
            update={"_next": NextNode.LOAD_INFERENCE},
        )

    async def _gate_feature_translation(self, state: Dict[str, Any]) -> Command:
        """
        Validate ml_input_vector output from feature_translation.

        Abort:
          required_tensile_strength_MPa > MAX_REALISTIC_STRENGTH_MPA
          -> geometry artefact (t_min fallback too small)

        Warnings (proceed):
          stiffness_dominated == True
          buckling_risk == True
        """
        vec      = state.get("ml_input_vector", {})
        warnings = list(state.get("supervisor_warnings", []))

        strength = vec.get("required_tensile_strength_MPa", 0.0) or 0.0

        # ── Abort: unrealistic strength requirement ─────────────────────
        if strength > self.thresholds.MAX_REALISTIC_STRENGTH_MPA:
            return await self._abort(
                stage=PipelineStage.FEATURE_TRANSLATION,
                trigger=(
                    f"required_tensile_strength_MPa = {strength:.0f} > "
                    f"{self.thresholds.MAX_REALISTIC_STRENGTH_MPA:.0f} "
                    "(exceeds any available engineering material — likely a "
                    "geometry artefact from a near-zero wall thickness estimate)"
                ),
                state_snapshot={
                    "required_tensile_strength_MPa": strength,
                    "min_wall_thickness_mm": state.get("geometry", {}).get(
                        "min_wall_thickness_mm"
                    ),
                    "is_watertight": state.get("geometry", {}).get("is_watertight"),
                    "primary_stress_mode": state.get("load_estimate", {}).get(
                        "primary_stress_mode"
                    ),
                },
                state=state,
            )

        # ── Warn: stiffness dominated ──────────────────────────────────
        if vec.get("stiffness_dominated"):
            warnings.append(
                f"Required stiffness ({vec.get('required_stiffness_GPa', '?')} GPa) "
                "exceeds any available material. Stiffness is geometry-constrained. "
                "The material recommendation optimises eco score within the "
                "stiffest available candidates, but a geometry change "
                "(thicker walls, shorter span, ribbing) is needed to truly "
                "meet the stiffness requirement."
            )

        # ── Warn: buckling risk ────────────────────────────────────────
        if vec.get("buckling_risk"):
            warnings.append(
                "Euler buckling analysis flagged this part as at risk under "
                "the inferred compression load. Consider adding lateral support "
                "or increasing the cross-section before committing to a material."
            )

        return Command(
            goto=NextNode.ML_PREDICTOR,
            update={
                "_next": NextNode.ML_PREDICTOR,
                "supervisor_warnings": warnings,
            },
        )

    async def _gate_ml_predictor(self, state: Dict[str, Any]) -> Command:
        """
        Validate predictions from ml_predictor.

        Abort:
          predictions is empty or None

        Warnings (proceed):
          all confidences < MIN_PREDICTION_CONFIDENCE
          top eco_score < MIN_ECO_SCORE_FOR_GOOD_FIT
        """
        predictions = state.get("predictions") or []
        warnings    = list(state.get("supervisor_warnings", []))

        # ── Abort: no predictions at all ──────────────────────────────
        if not predictions:
            return await self._abort(
                stage=PipelineStage.ML_PREDICTOR,
                trigger="predictions list is empty — ML model returned no candidates",
                state_snapshot={
                    "required_tensile_strength_MPa": state.get(
                        "ml_input_vector", {}
                    ).get("required_tensile_strength_MPa"),
                    "required_stiffness_GPa": state.get(
                        "ml_input_vector", {}
                    ).get("required_stiffness_GPa"),
                },
                state=state,
            )

        # ── Warn: low model confidence across the board ────────────────
        confidences = [p.get("confidence", 0.0) for p in predictions]
        if confidences and max(confidences) < self.thresholds.MIN_PREDICTION_CONFIDENCE:
            warnings.append(
                f"ML model confidence is low across all candidates "
                f"(max: {max(confidences):.2f}). The input vector may be "
                "outside the model's training distribution. "
                "Treat the recommendation as indicative rather than definitive."
            )

        # ── Warn: best eco score below threshold ───────────────────────
        top_eco = predictions[0].get("eco_score", 1.0)
        if top_eco < self.thresholds.MIN_ECO_SCORE_FOR_GOOD_FIT:
            warnings.append(
                f"The highest eco score among candidates is {top_eco:.2f} "
                f"(threshold: {self.thresholds.MIN_ECO_SCORE_FOR_GOOD_FIT:.2f}). "
                "No strongly eco-friendly material was found that meets the "
                "mechanical requirements. Consider relaxing the load constraints "
                "or revising the design to allow lighter-duty materials."
            )

        return Command(
            goto=NextNode.EXPLANATION,
            update={
                "_next": NextNode.EXPLANATION,
                "supervisor_warnings": warnings,
            },
        )

    async def _gate_explanation(self, state: Dict[str, Any]) -> Command:
        """
        After explanation, always route to report_assembly.
        Explanation handles its own retry internally.
        """
        return Command(
            goto=NextNode.REPORT_ASSEMBLY,
            update={"_next": NextNode.REPORT_ASSEMBLY},
        )

    # =======================================================================
    # Abort — the only place the LLM is called
    # =======================================================================

    async def _abort(
        self,
        stage: str,
        trigger: str,
        state_snapshot: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Command:
        """
        Generate a user-facing AbortReason via the LLM and route to END.

        The LLM produces a structured AbortReason (summary, root_cause,
        recommended_action). We store it in state["pipeline_error"] so
        report_assembly (or the caller) can surface it cleanly.

        If the LLM call itself fails (network error, timeout), we fall back
        to a deterministic error dict rather than raising — the pipeline
        should always reach END gracefully.
        """
        user_message = SupervisorPromptBuilder.build_abort_message(
            stage=stage,
            trigger=trigger,
            state_snapshot=state_snapshot,
        )

        try:
            abort_reason: AbortReason = await self.generate(
                messages=[{"role": "user", "content": user_message}],
                response_model=AbortReason,
            )
            error_dict = abort_reason.model_dump()

        except Exception as llm_error:
            # LLM unavailable — use a deterministic fallback so the
            # pipeline always terminates cleanly.
            error_dict = {
                "summary": f"Pipeline aborted at {stage}.",
                "root_cause": trigger,
                "recommended_action": (
                    "Review the CAD file and object description, then retry. "
                    f"(Error message generation also failed: {llm_error})"
                ),
            }

        return Command(
            goto=NextNode.END,
            update={
                "_next": NextNode.END,
                "pipeline_error": error_dict,
            },
        )
