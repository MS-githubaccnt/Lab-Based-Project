"""
supervisor/prompt_builder.py
-----------------------------
Prompt construction for the supervisor node.

The supervisor uses the LLM in exactly ONE scenario:
  generating a user-facing AbortReason when the pipeline cannot proceed.

All routing decisions are deterministic — no LLM needed there.
The system prompt therefore focuses entirely on producing clear,
actionable abort messages for a design engineer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePromptBuilder(ABC):
    @staticmethod
    @abstractmethod
    def build_main_prompt() -> str:
        pass


class SupervisorPromptBuilder(BasePromptBuilder):

    @staticmethod
    def build_main_prompt() -> str:
        """
        System prompt for abort message generation.
        Keeps the tone engineering-practical, not apologetic.
        """
        return """You are a mechanical engineering pipeline assistant. \
A material selection pipeline has encountered a condition it cannot recover from \
and must abort. Your job is to explain what happened and what the engineer \
should do next.

━━━ TONE ━━━
  - Direct and technical, but readable by a product design engineer
  - No apologies, no 'unfortunately'
  - Actionable: every response ends with a concrete step the user can take
  - Short: three fields, one or two sentences each

━━━ RULES ━━━
  - Reference the specific values that caused the abort (numbers, field names)
  - If the issue is in the CAD file: say what to change in the CAD tool
  - If the issue is in the object description: say what context to add
  - If the issue is a geometry artefact: explain the likely modelling error
  - Do NOT suggest contacting support — give the engineer something to do now"""

    @staticmethod
    def build_abort_message(
        stage: str,
        trigger: str,
        state_snapshot: Dict[str, Any],
    ) -> str:
        """
        Build the user message for abort LLM call.

        Args:
            stage:          The pipeline stage where the abort was triggered.
            trigger:        The specific condition that caused the abort.
                            e.g. 'dominant_axis == degenerate',
                                 'required_tensile_strength_MPa = 6200 > 5000'
            state_snapshot: Relevant subset of GraphState for context.
                            Only include fields relevant to this stage.

        Returns:
            Formatted string for the 'user' role message.
        """
        lines = [
            f"## Pipeline abort",
            f"  Stage:    {stage}",
            f"  Trigger:  {trigger}",
            "",
            "## Relevant context",
        ]

        for key, val in state_snapshot.items():
            if val is not None:
                lines.append(f"  {key}: {val}")

        lines += [
            "",
            "Produce an AbortReason explaining what went wrong "
            "and what the engineer should do to fix it.",
        ]

        return "\n".join(lines)
