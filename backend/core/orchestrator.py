"""
orchestrator.py
----------------
EcoMaterialOrchestrator: builds and runs the eco-material selection pipeline.

Public API
----------
process_request(...)
    Start a new pipeline run.

resume_with_clarification(session_id, answer, ...)
    Continue a run paused for clarification.

ask_followup(session_id, question, ...)
    Ask a follow-up question about the completed recommendation.
    Handled exclusively by ExplanationNode — no pipeline re-run.

Progress callbacks
------------------
progress_callback: Optional[Callable[[str, str, List[Thought]], Awaitable[None]]]
Called after each node with (node_name, status_message, new_thoughts).
Wire to Redis/WebSocket/logging. Default: None (silent).

Thoughts
--------
Every node emits Thought dicts describing what it computed. These accumulate
in GraphState['thoughts'] and are fired via progress_callback after each node.
The full thought trace is also included in the final report.
"""

from __future__ import annotations

import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver  # type: ignore
from langgraph.graph import END, StateGraph          # type: ignore

from agents.explanation.node import ExplanationNode
from agents.load_inference.node import LoadInferenceNode
from .stub_nodes import (
    ClarificationNode,
    MLPredictorNode,
    ReportAssemblyNode,
)
from .tool_nodes import CadParserNode, FeatureTranslationNode
from schema.state import GraphState, PipelineResult, Thought
from agents.supervisor.node import SupervisorNode
from agents.supervisor.schemas import NextNode

# progress_callback signature: (node_name, status_message, new_thoughts)
ProgressCallback = Optional[
    Callable[[str, str, List[Thought]], Awaitable[None]]
]

_NODE_STATUS: Dict[str, str] = {
    "supervisor":           "Analysing pipeline state...",
    "cad_parser":           "Parsing CAD file and extracting geometry...",
    "load_inference":       "Inferring load profile from object description...",
    "clarification":        "Waiting for clarification...",
    "feature_translation":  "Deriving mechanical requirements...",
    "ml_predictor":         "Running material selection model...",
    "explanation":          "Generating recommendation explanation...",
    "report_assembly":      "Assembling final report...",
}


class EcoMaterialOrchestrator:

    def __init__(self) -> None:
        self._checkpointer   = MemorySaver()

        # Hold ExplanationNode instance so ask_followup can reuse it
        self._explanation_node = ExplanationNode()
        self._graph          = self._build_graph()

    # =======================================================================
    # Graph construction
    # =======================================================================

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        supervisor           = SupervisorNode()
        cad_parser           = CadParserNode()
        load_inference       = LoadInferenceNode()
        clarification        = ClarificationNode()
        feature_translation  = FeatureTranslationNode()
        ml_predictor         = MLPredictorNode()
        explanation          = self._explanation_node
        report_assembly      = ReportAssemblyNode()

        workflow.add_node("supervisor",          supervisor.invoke)
        workflow.add_node("cad_parser",          cad_parser.invoke)
        workflow.add_node("load_inference",      load_inference.invoke)
        workflow.add_node("clarification",       clarification.invoke)
        workflow.add_node("feature_translation", feature_translation.invoke)
        workflow.add_node("ml_predictor",        ml_predictor.invoke)
        workflow.add_node("explanation",         explanation.invoke)
        workflow.add_node("report_assembly",     report_assembly.invoke)

        workflow.set_entry_point("supervisor")

        # Supervisor routes forward based on state['_next']
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state.get("_next", NextNode.END),
            {
                "cad_parser":          "cad_parser",
                "load_inference":      "load_inference",
                "clarification":       "clarification",
                "feature_translation": "feature_translation",
                "ml_predictor":        "ml_predictor",
                "explanation":         "explanation",
                "report_assembly":     "report_assembly",
                NextNode.END:          END,
            },
        )

        # All content nodes (except report_assembly) loop back to supervisor
        for name in [
            "cad_parser", "load_inference", "clarification",
            "feature_translation", "ml_predictor", "explanation",
        ]:
            workflow.add_edge(name, "supervisor")

        # report_assembly goes straight to END — no quality gate needed
        workflow.add_edge("report_assembly", END)

        return workflow.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["clarification"],
        )

    # =======================================================================
    # Public API
    # =======================================================================

    async def process_request(
        self,
        cad_file_path: str,
        object_description: str,
        session_id: Optional[str] = None,
        recyclability_priority: float = 0.7,
        progress_callback: ProgressCallback = None,
    ) -> PipelineResult:
        """
        Start a new eco-material selection pipeline run.

        Args:
            cad_file_path:          Path to the .step, .stp, or .stl file.
            object_description:     Plain-language part description.
            session_id:             Optional. Auto-generated if not provided.
                                    Required for resume/followup calls.
            recyclability_priority: Eco weighting 0.0–1.0. Default 0.7.
            progress_callback:      Optional async fn(node, status, thoughts).

        Returns:
            PipelineResult with status 'complete', 'clarification_needed',
            or 'error'.
        """
        if not 0.0 <= recyclability_priority <= 1.0:
            raise ValueError(
                f"recyclability_priority must be 0.0–1.0, got {recyclability_priority}."
            )

        session_id = session_id or str(uuid.uuid4())
        config     = {"configurable": {"thread_id": session_id}}

        initial_state: GraphState = {
            "cad_file_path":          cad_file_path,
            "object_description":     object_description,
            "recyclability_priority": recyclability_priority,
            "pipeline_stage":         None,
            "supervisor_warnings":    [],
            "pipeline_error":         None,
            "load_inference_retries": 0,
            "_next":                  "",
            "thoughts":               [],
            "conversation_history":   [],
        }

        final_state = await self._run_graph(initial_state, config, progress_callback)
        return self._build_result(final_state, session_id)

    async def resume_with_clarification(
        self,
        session_id: str,
        clarification_answer: str,
        progress_callback: ProgressCallback = None,
    ) -> PipelineResult:
        """
        Resume a pipeline paused for clarification.

        Args:
            session_id:           From the original process_request() call.
            clarification_answer: User's answer to the clarification question.
            progress_callback:    Optional async fn(node, status, thoughts).

        Returns:
            PipelineResult — typically 'complete' or 'error'.
        """
        config = {"configurable": {"thread_id": session_id}}

        self._graph.update_state(
            config,
            {"clarification_answer": clarification_answer},
        )

        final_state = await self._run_graph(None, config, progress_callback)
        return self._build_result(final_state, session_id)

    async def ask_followup(
        self,
        session_id: str,
        question: str,
        progress_callback: ProgressCallback = None,
    ) -> PipelineResult:
        """
        Ask a follow-up question about the completed recommendation.

        Does NOT re-run the full pipeline. Calls ExplanationNode.handle_followup()
        directly with the full pipeline context from the checkpoint.

        Args:
            session_id: From the original process_request() call.
                        Must be a completed run (status='complete').
            question:   The engineer's follow-up question.
            progress_callback: Optional async fn(node, status, thoughts).

        Returns:
            PipelineResult(status='followup_answered', followup_answer=...)
        """
        config = {"configurable": {"thread_id": session_id}}

        # Read current state from checkpoint
        checkpoint = self._graph.get_state(config)
        if not checkpoint or not checkpoint.values:
            raise ValueError(
                f"No checkpoint found for session_id='{session_id}'. "
                "Ensure process_request() completed successfully before "
                "calling ask_followup()."
            )

        current_state = dict(checkpoint.values)

        if not current_state.get("pipeline_context"):
            raise ValueError(
                f"Session '{session_id}' has no pipeline_context. "
                "ask_followup() requires a completed pipeline run "
                "(status='complete')."
            )

        # Fire pre-call progress update
        if progress_callback:
            try:
                await progress_callback(
                    "explanation",
                    "Answering follow-up question...",
                    [],
                )
            except Exception:
                pass

        # ExplanationNode handles the follow-up directly
        state_delta = await self._explanation_node.handle_followup(
            state=current_state,
            question=question,
        )

        # Persist the updated conversation_history and thoughts to the checkpoint
        self._graph.update_state(config, state_delta, as_node="explanation")

        answer     = state_delta.get("followup_answer", "")
        new_thoughts = state_delta.get("thoughts", [])[len(current_state.get("thoughts") or []):]

        # Fire post-call progress update with the new thought
        if progress_callback and new_thoughts:
            try:
                await progress_callback("explanation", "Follow-up answered.", new_thoughts)
            except Exception:
                pass

        return PipelineResult(
            status="followup_answered",
            session_id=session_id,
            followup_answer=answer,
            warnings=current_state.get("supervisor_warnings") or [],
            thoughts=new_thoughts,
        )

    # =======================================================================
    # Internal helpers
    # =======================================================================

    async def _run_graph(
        self,
        state: Optional[GraphState],
        config: Dict[str, Any],
        progress_callback: ProgressCallback,
    ) -> Dict[str, Any]:
        """
        Stream graph events, fire progress callbacks with thoughts, and
        return the final merged state.
        """
        final_state: Dict[str, Any] = {}
        # Track which thoughts were present before each node runs so we can
        # extract only the NEW thoughts emitted by each node.
        prev_thought_count = 0

        async for event in self._graph.astream(state, config=config):
            node_name   = next(iter(event))
            node_output = event[node_name]

            if isinstance(node_output, dict):
                final_state.update(node_output)

            if progress_callback and node_name in _NODE_STATUS:
                # Extract only the thoughts this node added
                all_thoughts   = final_state.get("thoughts") or []
                new_thoughts   = all_thoughts[prev_thought_count:]
                prev_thought_count = len(all_thoughts)

                try:
                    await progress_callback(
                        node_name,
                        _NODE_STATUS[node_name],
                        new_thoughts,
                    )
                except Exception:
                    pass  # progress callback failure must never crash the pipeline

        return final_state

    def _build_result(
        self,
        state: Dict[str, Any],
        session_id: str,
    ) -> PipelineResult:
        """
        Inspect final state and return the appropriate PipelineResult.
        """
        warnings = state.get("supervisor_warnings") or []
        thoughts = state.get("thoughts") or []

        # Aborted
        if state.get("pipeline_error"):
            return PipelineResult(
                status="error",
                session_id=session_id,
                error=state["pipeline_error"],
                warnings=warnings,
                thoughts=thoughts,
            )

        # Paused for clarification
        if (
            state.get("clarification_question")
            and not state.get("clarification_answer")
            and not state.get("report")
        ):
            return PipelineResult(
                status="clarification_needed",
                session_id=session_id,
                clarification_question=state["clarification_question"],
                warnings=warnings,
                thoughts=thoughts,
            )

        # Completed
        if state.get("report"):
            return PipelineResult(
                status="complete",
                session_id=session_id,
                report=state["report"],
                warnings=warnings,
                thoughts=thoughts,
            )

        # Unexpected terminal state
        return PipelineResult(
            status="error",
            session_id=session_id,
            error={
                "summary": "Pipeline completed without producing a report.",
                "root_cause": (
                    f"pipeline_stage={state.get('pipeline_stage')}, "
                    f"report={state.get('report')}, "
                    f"pipeline_error={state.get('pipeline_error')}."
                ),
                "recommended_action": "Check node logs for the last completed stage.",
            },
            warnings=warnings,
            thoughts=thoughts,
        )