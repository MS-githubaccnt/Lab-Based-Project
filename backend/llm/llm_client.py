import os
from typing import Any, Dict, List, Optional, Type
from groq import AsyncGroq
from pydantic import BaseModel
from .structured_outputs import StructuredOutputEnforcer
from config.settings import settings
class GroqLLMClient:
    """
    Async Groq LLM wrapper supporting both raw text and strict Pydantic outputs.
    Optimized for multi-agent message history (Episodic Memory).
    """

    def __init__(
        self,
        model: str = "llama3-70b-8192",  # Defaulting to a capable Groq model
        temperature: float = 0.0,        # 0.0 is best for data extraction/routing
        top_p: float = 0.9,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set. Check your .env file.")

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        
        # Initialize the base async client once
        self._base_client = AsyncGroq(api_key=self.api_key)

    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        response_model: Optional[Type[BaseModel]] = None
    ) -> Any:
        """
        Generates a response from the LLM.
        
        Args:
            messages: A list of dicts (role/content) representing the conversation history.
            response_model: Optional Pydantic model. If provided, returns a parsed Pydantic object. 
                            If None, returns raw string content.
        """
        
        if self.system_prompt and not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Path A: Structured Pydantic Output (Used by Supervisor/Context Agents)
        if response_model:
            enforcer = StructuredOutputEnforcer(response_model)
            instructor_client = enforcer.wrap(self._base_client)
            
            response = await instructor_client.chat.completions.create(
                model=self.model,
                response_model=response_model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return response

        # Path B: Standard Text Output (Used for final synthesis/general chat)
        completion = await self._base_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return completion.choices[0].message.content