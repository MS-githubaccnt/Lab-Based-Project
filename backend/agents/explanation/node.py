"""
explanation/node.py
--------------------
ExplanationNode: generates the initial recommendation explanation
and handles all follow-up questions from the engineer.

Initial run (pipeline mode)
----------------------------
invoke(state) is called by LangGraph after ml_predictor completes.
It generates a three-paragraph grounded explanation, emits thoughts
showing what it is doing, stores the explanation and pipeline_context
in state, and seeds the conversation_history with the explanation as
the first assistant turn.

Follow-up mode
--------------
handle_followup(state, question) is called directly by the orchestrator's
ask_followup() method — NOT through the LangGraph graph. It appends the
user question to conversation_history, calls the LLM with full context,
appends the answer, and returns the updated state delta.

Thoughts
--------
The node emits thoughts: List[Thought] describing what it is doing and
what it found. These are accumulated in GraphState['thoughts'] and
streamed to the frontend via the progress_callback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .prompt_builder import ExplanationPromptBuilder
from schema.state import Thought
from agents.base_agent import BaseAgentNode


DEFAULT_MODEL       = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2
_MAX_RETRIES        = 1

# Maximum conversation turns kept in history to avoid context overflow.
# Older turns are trimmed from the middle, keeping the first turn
# (initial explanation) and the most recent N turns.
_MAX_HISTORY_TURNS = 10


class ExplanationNode(BaseAgentNode):
    """
    Generates the initial recommendation explanation and handles follow-up
    questions. The only node in the pipeline that maintains a conversation.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            system_prompt=ExplanationPromptBuilder.build_main_prompt(),
        )

    # =======================================================================
    # LangGraph node — initial explanation
    # =======================================================================

    async def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate the initial explanation for the current recommendation.

        Reads:  object_description, geometry, load_estimate,
                ml_input_vector, predictions, clarification_answer (optional)
        Writes: explanation, pipeline_context, conversation_history,
                thoughts (appended), pipeline_stage
        """
        object_description = self._require(state, "object_description", str)
        geometry           = self._require(state, "geometry",           dict)
        load_estimate      = self._require(state, "load_estimate",      dict)
        ml_input_vector    = self._require(state, "ml_input_vector",    dict)
        predictions        = self._require(state, "predictions",        list)
        clarification      = state.get("clarification_answer")

        existing_thoughts  = list(state.get("thoughts") or [])
        node               = "explanation"

        # ── Opening thought ───────────────────────────────────────────────
        new_thoughts: List[Thought] = [
            _thought(node, "info", "Generating grounded recommendation explanation...")
        ]

        if not predictions:
            fallback = (
                "No material predictions were returned. "
                "Check that the ML predictor ran successfully."
            )
            new_thoughts.append(
                _thought(node, "warning", "No predictions available — cannot explain.")
            )
            return {
                "explanation":          {"error": fallback},
                "pipeline_context":     None,
                "conversation_history": [],
                "thoughts":             existing_thoughts + new_thoughts,
                "pipeline_stage":       "explanation",
            }

        # ── Build initial message ─────────────────────────────────────────
        user_message = ExplanationPromptBuilder.build_user_message(
            object_description=object_description,
            geometry=geometry,
            load_estimate=load_estimate,
            ml_input_vector=ml_input_vector,
            predictions=predictions,
            clarification_answer=clarification,
        )
        messages = [{"role": "user", "content": user_message}]

        # ── Generate ──────────────────────────────────────────────────────
        explanation_text = await self._generate_text(messages)

        # ── Emit result thoughts ──────────────────────────────────────────
        top = predictions[0]
        new_thoughts.append(_thought(
            node, "result",
            f"Top material: {top.get('material_name', '?')} "
            f"(eco score {top.get('eco_score', '?'):.2f})"
        ))
        if len(predictions) > 1:
            runner = predictions[1]
            new_thoughts.append(_thought(
                node, "result",
                f"Runner-up: {runner.get('material_name', '?')} "
                f"(eco score {runner.get('eco_score', '?'):.2f})"
            ))
        new_thoughts.append(
            _thought(node, "info", "Explanation complete. Ready for follow-up questions.")
        )

        # ── Build pipeline_context for follow-up use ──────────────────────
        pipeline_context = _build_pipeline_context(
            object_description, geometry, load_estimate, ml_input_vector, predictions
        )

        # ── Seed conversation_history with the initial explanation ─────────
        conversation_history = [
            {"role": "assistant", "content": explanation_text}
        ]

        return {
            "explanation":          {"text": explanation_text},
            "pipeline_context":     pipeline_context,
            "conversation_history": conversation_history,
            "thoughts":             existing_thoughts + new_thoughts,
            "pipeline_stage":       "explanation",
        }

    # =======================================================================
    # Follow-up mode — called directly by orchestrator, not via graph
    # =======================================================================

    async def handle_followup(
        self,
        state: Dict[str, Any],
        question: str,
    ) -> Dict[str, Any]:
        """
        Answer a follow-up question using full pipeline context and history.

        Called directly by EcoMaterialOrchestrator.ask_followup().
        Does NOT route through the LangGraph graph.

        Args:
            state:    The current checkpoint state (contains pipeline_context
                      and conversation_history from the initial run).
            question: The engineer's follow-up question.

        Returns:
            State delta: {
                conversation_history: updated list,
                followup_answer:      the answer string,
                thoughts:             one new thought
            }
        """
        question = question.strip()
        if not question:
            raise ValueError("Follow-up question must not be empty.")

        pipeline_context    = state.get("pipeline_context")
        conversation_history = list(state.get("conversation_history") or [])

        if not pipeline_context:
            raise ValueError(
                "pipeline_context is missing from state. "
                "ask_followup() requires a completed pipeline run "
                "(process_request() must have returned status='complete' first)."
            )

        if not conversation_history:
            raise ValueError(
                "conversation_history is empty — the initial explanation "
                "has not been generated yet. "
                "Ensure process_request() completed successfully before calling ask_followup()."
            )

        # ── Build messages: context injection + history + new question ─────
        messages = _build_followup_messages(
            pipeline_context=pipeline_context,
            conversation_history=conversation_history,
            question=question,
        )

        # ── Generate answer ───────────────────────────────────────────────
        answer = await self._generate_text(messages)

        # ── Append to history ─────────────────────────────────────────────
        conversation_history.append({"role": "user",      "content": question})
        conversation_history.append({"role": "assistant", "content": answer})

        # ── Trim history if it's growing too long ─────────────────────────
        conversation_history = _trim_history(conversation_history, _MAX_HISTORY_TURNS)

        # ── Emit one thought ──────────────────────────────────────────────
        existing_thoughts = list(state.get("thoughts") or [])
        new_thought = _thought(
            "explanation", "info",
            f"Follow-up answered ({len(conversation_history) // 2} turns so far)"
        )

        return {
            "conversation_history": conversation_history,
            "followup_answer":      answer,
            "thoughts":             existing_thoughts + [new_thought],
        }

    # =======================================================================
    # Internal helpers
    # =======================================================================

    async def _generate_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM and return a plain string response.
        No response_model — free prose output.
        """
        response = await self.generate(messages=messages)
        return _extract_text(response)

    @staticmethod
    def _require(state: Dict[str, Any], key: str, expected_type: type) -> Any:
        if key not in state:
            raise KeyError(
                f"ExplanationNode requires state['{key}'] but it was not found. "
                f"Ensure the preceding node wrote '{key}' to state."
            )
        value = state[key]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"state['{key}'] must be {expected_type.__name__}, "
                f"got {type(value).__name__}."
            )
        return value


# ===========================================================================
# Module-level helpers
# ===========================================================================

def _thought(node: str, type_: str, text: str) -> Thought:
    return Thought(node=node, type=type_, text=text)


def _extract_text(response: Any) -> str:
    """Normalise LLM response to a plain string."""
    if isinstance(response, str):
        return response.strip()
    if hasattr(response, "content"):
        return str(response.content).strip()
    if hasattr(response, "text"):
        return str(response.text).strip()
    return str(response).strip()


def _build_pipeline_context(
    object_description: str,
    geometry: Dict[str, Any],
    load_estimate: Dict[str, Any],
    ml_input_vector: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Store the minimal subset of pipeline data needed to answer follow-ups.
    Avoids storing the full state (which includes raw mesh data, etc.).
    """
    return {
        "object_description": object_description,
        "geometry": {
            k: geometry.get(k) for k in (
                "bbox_x_mm", "bbox_y_mm", "bbox_z_mm",
                "dominant_axis", "aspect_ratio",
                "min_wall_thickness_mm", "median_wall_thickness_mm",
                "has_thin_features", "is_watertight", "volume_mm3",
            )
        },
        "load_estimate": {
            k: load_estimate.get(k) for k in (
                "load_type", "primary_stress_mode", "magnitude_range_N",
                "safety_factor", "operating_temp_C", "is_fatigue_critical",
                "deflection_category", "reasoning",
            )
        },
        "ml_input_vector": {
            k: ml_input_vector.get(k) for k in (
                "required_tensile_strength_MPa", "required_stiffness_GPa",
                "stiffness_dominated", "buckling_risk", "fatigue_critical",
                "max_density_kg_m3", "min_operating_temp_C", "max_operating_temp_C",
                "derivation",
            )
        },
        "predictions": predictions[:3],
    }


def _build_followup_messages(
    pipeline_context: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    question: str,
) -> List[Dict[str, str]]:
    """
    Build the messages list for a follow-up LLM call.

    Structure:
      1. Context injection as a 'user' turn (pipeline data fact sheet)
      2. Acknowledgement as 'assistant' turn (keeps the conversation clean)
      3. Full conversation history (initial explanation + any prior Q&As)
      4. New question as the final 'user' turn

    The context injection is placed BEFORE the history so the LLM always
    has access to the raw numbers regardless of how long the conversation has grown.
    """
    from explanation.prompt_builder import ExplanationPromptBuilder

    ctx = pipeline_context
    context_block = ExplanationPromptBuilder.build_context_injection(
        object_description=ctx["object_description"],
        geometry=ctx["geometry"],
        load_estimate=ctx["load_estimate"],
        ml_input_vector=ctx["ml_input_vector"],
        predictions=ctx["predictions"],
    )

    messages = [
        {"role": "user",      "content": context_block},
        {"role": "assistant", "content": "Understood. I have the full pipeline context. Ready for your question."},
        *conversation_history,
        {"role": "user",      "content": question},
    ]

    return messages


def _trim_history(
    history: List[Dict[str, str]],
    max_turns: int,
) -> List[Dict[str, str]]:
    """
    Trim conversation history to at most max_turns Q&A pairs.
    Always preserves the first entry (initial explanation).
    Drops the oldest Q&A pairs from the middle when over budget.

    A 'turn' = one user message + one assistant message = 2 entries.
    """
    if len(history) <= max_turns * 2 + 1:
        return history

    # Keep: first entry (initial explanation) + most recent (max_turns - 1) Q&A pairs
    first      = history[:1]
    recent_qa  = history[-(max_turns - 1) * 2:]
    return first + recent_qa