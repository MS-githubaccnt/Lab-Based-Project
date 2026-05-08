"""
schema/state.py
----------------
GraphState: the single TypedDict passed through every node.
PipelineResult: the structured return value of the orchestrator's public API.
Thought: typed dict for a single agent reasoning step shown on the frontend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Thought — a single reasoning step emitted by a node
# ---------------------------------------------------------------------------

class Thought(TypedDict):
    """
    A single visible reasoning step emitted by a pipeline node.
    The frontend uses 'type' to style the entry:
      'info'    -> neutral / grey   ("Parsing STEP file...")
      'result'  -> green            ("Required strength: 1,859 MPa")
      'warning' -> amber            ("Mesh not watertight — estimates approximate")
    """
    node: str
    type: Literal["info", "result", "warning"]
    text: str


# ---------------------------------------------------------------------------
# GraphState
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    Complete state for the eco-material selection pipeline.
    total=False: all fields optional at TypedDict level; nodes enforce
    required fields themselves via _require() helpers at runtime.
    """

    # ── Caller-supplied inputs ────────────────────────────────────────────
    cad_file_path:           str
    object_description:      str
    recyclability_priority:  float
    expected_load_n:         float
    safety_factor:           float

    # ── Supervisor bookkeeping ────────────────────────────────────────────
    pipeline_stage:          str
    supervisor_warnings:     List[str]
    pipeline_error:          Optional[Dict[str, Any]]
    load_inference_retries:  int
    _next:                   str

    # ── Thoughts (streaming reasoning shown on frontend) ──────────────────
    thoughts:                List[Thought]
    """
    Accumulated list of Thought dicts from all nodes that have run.
    Each node appends its own thoughts — never overwrites.
    The orchestrator fires progress_callback with the latest batch
    after each node completes.
    """

    # ── CadParser outputs ─────────────────────────────────────────────────
    geometry:                Optional[Dict[str, Any]]

    # ── LoadInference outputs ─────────────────────────────────────────────
    load_estimate:           Optional[Dict[str, Any]]
    inference_confidence:    Optional[float]
    load_inference_hint:     Optional[str]

    # ── FeatureTranslation outputs ────────────────────────────────────────
    ml_input_vector:         Optional[Dict[str, Any]]

    # ── MLPredictor outputs ───────────────────────────────────────────────
    predictions:             Optional[List[Dict[str, Any]]]

    # ── Explanation outputs ───────────────────────────────────────────────
    explanation:             Optional[Dict[str, Any]]

    pipeline_context:        Optional[Dict[str, Any]]
    """
    Scoped snapshot of pipeline state written by ExplanationNode after the
    initial explanation. Contains the data needed to answer follow-up
    questions: object_description, geometry summary, load_estimate,
    ml_input_vector, top predictions.
    Read by ask_followup() via the checkpoint.
    """

    conversation_history:    List[Dict[str, str]]
    """
    Chat history for follow-up questions handled by ExplanationNode.
    Format: [{'role': 'assistant'|'user', 'content': str}, ...]
    Starts empty. ExplanationNode appends its initial explanation as
    the first assistant turn. Each ask_followup() call appends
    the user question and the LLM answer.
    """

    # ── ReportAssembly outputs ────────────────────────────────────────────
    report:                  Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

class PipelineResult:
    """
    Return value from process_request() and ask_followup().

    status='complete'             -> report is populated
    status='followup_answered'    -> followup_answer populated
    status='error'                -> error populated with AbortReason
    """

    __slots__ = (
        "status", "session_id", "report",
        "warnings", "error", "followup_answer", "thoughts",
    )

    def __init__(
        self,
        status: Literal["complete", "followup_answered", "error"],
        session_id: str,
        report: Optional[Dict[str, Any]] = None,
        followup_answer: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        error: Optional[Dict[str, Any]] = None,
        thoughts: Optional[List[Thought]] = None,
    ) -> None:
        self.status                 = status
        self.session_id             = session_id
        self.report                 = report
        self.followup_answer        = followup_answer
        self.warnings               = warnings or []
        self.error                  = error
        self.thoughts               = thoughts or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status":                 self.status,
            "session_id":             self.session_id,
            "report":                 self.report,
            "followup_answer":        self.followup_answer,
            "warnings":               self.warnings,
            "error":                  self.error,
            "thoughts":               self.thoughts,
        }

    def __repr__(self) -> str:
        return (
            f"PipelineResult(status={self.status!r}, "
            f"session_id={self.session_id!r}, "
            f"thoughts={len(self.thoughts)}, "
            f"warnings={len(self.warnings)})"
        )
