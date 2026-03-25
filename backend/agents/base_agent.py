from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import logging
from llm.llm_client import GroqLLMClient

logger = logging.getLogger("roster-neural")
class BaseAgentNode(ABC):
    """
    Base class for all Agent Nodes.
    Provides a shared GroqLLMClient wrapper and a helper `generate` method,
    and enforces an async `invoke` interface.
    """

    def __init__(
        self,
        model: str,
        temperature: float,
        system_prompt: str,
    ) -> None:
        self.llm = GroqLLMClient(
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
        )

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        response_model: Optional[Type[Any]] = None,
        tools: Optional[List[Any]] = None,
    ) -> Any:
        """
        Thin wrapper over GroqLLMClient.generate so subclasses don't need
        to touch the underlying client directly.
        """
        kwargs: Dict[str, Any] = {"messages": messages}
        if response_model is not None:
            kwargs["response_model"] = response_model
        if tools is not None:
            kwargs["tools"] = tools
            
        tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools] if tools else "None"
        logger.info(f"[Agent] Generating LLM response. Tools available: {tool_names}")
        
        return await self.llm.generate(**kwargs)

    @abstractmethod
    async def invoke(self, state: Any) -> Dict[str, Any]:
        """
        ABSTRACT: Execute the agent node given the current state and
        return a state delta / update.
        Every concrete AgentNode must implement this method.
        """
        raise NotImplementedError