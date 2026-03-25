import instructor
from groq import AsyncGroq
from typing import Type
from pydantic import BaseModel

class StructuredOutputEnforcer:
    """
    Wraps an AsyncGroq LLM client and enforces structured Pydantic output 
    using modern instructor syntax.
    """
    def __init__(self, response_model: Type[BaseModel]):
        self.response_model = response_model

    def wrap(self, base_client: AsyncGroq) -> instructor.AsyncInstructor:
        """
        Wraps the base async LLM client with Instructor for Groq.
        Using Mode.JSON or Mode.TOOLS ensures robust JSON schema adherence.
        """
        return instructor.from_groq(base_client, mode=instructor.Mode.TOOLS)