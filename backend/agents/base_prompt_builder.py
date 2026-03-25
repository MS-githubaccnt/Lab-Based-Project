from abc import ABC, abstractmethod


class BasePromptBuilder(ABC):
    """
    Base class for all Prompt Builders.
    Enforces a common `build_main_prompt` interface.
    """

    @staticmethod
    @abstractmethod
    def build_main_prompt() -> str:
        """
        ABSTRACT: Build the main/system prompt string for an agent.
        """
        pass