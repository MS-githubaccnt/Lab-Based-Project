from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .prompt_builder import LoadInferencePromptBuilder
from .schemas import CONFIDENCE_THRESHOLD, LoadEstimate
from agents.base_agent import BaseAgentNode
from agents.supervisor.schemas import PipelineStage
from schema.state import Thought



DEFAULT_MODEL       = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2   # low temperature: we want consistent structured output,
                             # not creative variation in load estimates


class LoadInferenceNode(BaseAgentNode):
    """
    Agent node that infers a mechanical load profile from:
      - GraphState["object_description"]  — user's plain-language part name
      - GraphState["geometry"]            — GeometryFeatures dict from cad_parser
      - GraphState["clarification_answer"] — optional user reply (second pass)

    Writes to GraphState:
      - load_estimate            : LoadEstimate dict
      - inference_confidence     : float  (used by the conditional edge)
      - clarification_question   : str | None
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__(
            model=model,
            temperature=temperature,
            system_prompt=LoadInferencePromptBuilder.build_main_prompt(),
        )

    async def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute load inference against the current graph state.

        Args:
            state: Full GraphState dict. Must contain:
                   "object_description" (str)
                   "geometry" (dict, output of cad_parser node)
                   Optionally: "clarification_answer" (str)

        Returns:
            State delta dict with keys:
                load_estimate          : dict (LoadEstimate serialised)
                inference_confidence   : float
                clarification_question : str | None
        """
        # ── Read from state ───────────────────────────────────────────────
        object_description = self._require(state, "object_description", str)
        geometry           = self._require(state, "geometry", dict)
        clarification      = state.get("clarification_answer")  # may be None
        
        existing_thoughts = list(state.get("thoughts") or [])
        node = "load_inference"
        new_thoughts = [
            Thought(node=node, type="info", text="Inferring mechanical load profile...")
        ]

        # ── Build messages ────────────────────────────────────────────────
        user_message = LoadInferencePromptBuilder.build_user_message(
            object_description=object_description,
            geometry=geometry,
            clarification_answer=clarification,
        )

        messages = [{"role": "user", "content": user_message}]

        # ── Call LLM with structured output ───────────────────────────────
        # GroqLLMClient.generate() with response_model uses instructor
        # to enforce the LoadEstimate schema. The returned object is a
        # validated LoadEstimate pydantic instance.
        estimate: LoadEstimate = await self.generate(
            messages=messages,
            response_model=LoadEstimate,
        )

        # ── Return state delta ────────────────────────────────────────────
        new_thoughts.append(Thought(node=node, type="result", text="Load inference complete."))
        
        return {
            "load_estimate":          estimate.model_dump(mode="json"),
            "inference_confidence":   estimate.confidence,
            "clarification_question": estimate.clarification_question,
            "pipeline_stage":         PipelineStage.LOAD_INFERENCE,
            "thoughts":               existing_thoughts + new_thoughts,
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _require(state: Dict[str, Any], key: str, expected_type: type) -> Any:
        """
        Read a required key from state with type checking.

        Raises KeyError if missing, TypeError if wrong type.
        Clear error messages help debug misconfigured graph wiring.
        """
        if key not in state:
            raise KeyError(
                f"LoadInferenceNode.invoke() requires state['{key}'] "
                f"but it was not found. "
                f"Check that the preceding node writes '{key}' to state."
            )
        value = state[key]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"state['{key}'] must be {expected_type.__name__}, "
                f"got {type(value).__name__}. "
                f"Check the output of the preceding node."
            )
        return value
