"""
schemas/api_models.py
----------------------
Pydantic request and response models for the eco-material analysis API.

All request models validate inputs before the background task is queued,
so failures surface as HTTP 422 rather than silent async task errors.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Nested schema
# ---------------------------------------------------------------------------

class ThoughtSchema(BaseModel):
    """A single reasoning step emitted by a pipeline node."""
    node: str
    type: Literal["info", "result", "warning"]
    text: str


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class StartAnalysisRequest(BaseModel):
    """
    Payload for POST /api/v1/analysis/start.

    session_id is optional — the server generates one if absent and echoes
    it back in the response. The frontend must store it for follow-up calls.
    """
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session identifier. Generated server-side if not provided. "
            "Store this value — it is required for follow-up calls."
        ),
    )
    cad_file_path: str = Field(
        description=(
            "Absolute server-side path to the uploaded .step, .stp, or .stl file. "
            "This should be the path returned by your file upload endpoint."
        ),
    )
    object_description: str = Field(
        min_length=3,
        max_length=500,
        description=(
            "Plain-language description of the part. "
            "e.g. 'bicycle crank arm', 'structural drone arm bracket'. "
            "The more specific, the more accurate the load inference."
        ),
    )
    recyclability_priority: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Eco weighting for material selection. "
            "0.0 = performance only, 1.0 = maximum recyclability. "
            "Default 0.7."
        ),
    )

    @field_validator("cad_file_path")
    @classmethod
    def file_must_exist(cls, v: str) -> str:
        if not os.path.isfile(v):
            raise ValueError(
                f"CAD file not found on server: '{v}'. "
                "Ensure the file has been uploaded successfully before "
                "starting an analysis."
            )
        return v

    @field_validator("object_description")
    @classmethod
    def description_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "object_description must not be blank. "
                "Provide a plain-language description of the part, "
                "e.g. 'bicycle crank arm'."
            )
        return v.strip()


class FollowupRequest(BaseModel):
    """
    Payload for POST /api/v1/analysis/followup.
    Ask a follow-up question about a completed recommendation.
    The pipeline does not re-run — ExplanationNode answers directly.
    """
    session_id: str = Field(
        description="The session_id from the original completed analysis.",
    )
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="The engineer's follow-up question about the recommendation.",
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank.")
        return v.strip()


class ContinueAnalysisRequest(BaseModel):
    """
    Payload for POST /api/v1/analysis/continue.
    Starts the full material-selection pipeline after the CAD parser preview
    has been reviewed in the frontend.
    """
    upload_id: str = Field(
        min_length=1,
        description="Identifier returned by POST /api/v1/analysis/parse.",
    )
    object_description: str = Field(
        min_length=3,
        max_length=500,
        description="Plain-language description of the part.",
    )
    recyclability_priority: float = Field(default=0.7, ge=0.0, le=1.0)
    expected_load_n: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="User-approved expected operating load in Newtons.",
    )
    safety_factor: Optional[float] = Field(
        default=None,
        ge=1.0,
        description="User-approved factor of safety.",
    )

    @field_validator("object_description")
    @classmethod
    def continue_description_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("object_description must not be blank.")
        return v.strip()


class LoadPreviewRequest(BaseModel):
    """
    Payload for POST /api/v1/analysis/load-preview.
    Runs CAD parsing + LoadInferenceNode only, so the frontend can let the
    user review and edit expected load assumptions before material selection.
    """
    upload_id: str = Field(
        min_length=1,
        description="Identifier returned by POST /api/v1/analysis/parse.",
    )
    object_description: str = Field(
        min_length=3,
        max_length=500,
        description="Plain-language description of the part.",
    )

    @field_validator("object_description")
    @classmethod
    def preview_description_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("object_description must not be blank.")
        return v.strip()


class FeaturePreviewRequest(BaseModel):
    """
    Payload for POST /api/v1/analysis/feature-preview.
    Runs LoadInferenceNode with user-approved overrides, then runs the
    feature translator so the frontend can preview interpreted requirements.
    """
    upload_id: str = Field(
        min_length=1,
        description="Identifier returned by POST /api/v1/analysis/parse.",
    )
    object_description: str = Field(
        min_length=3,
        max_length=500,
        description="Plain-language description of the part.",
    )
    expected_load_n: float = Field(
        ge=0.0,
        description="User-approved expected operating load in Newtons.",
    )
    safety_factor: float = Field(
        ge=1.0,
        description="User-approved factor of safety.",
    )

    @field_validator("object_description")
    @classmethod
    def feature_description_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("object_description must not be blank.")
        return v.strip()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AcceptedResponse(BaseModel):
    """
    Returned immediately by start and followup routes.
    The frontend uses task_id to poll /status/{task_id}.
    session_id is echoed back so the frontend can store it for future calls.
    """
    status: Literal["accepted"] = "accepted"
    task_id: str = Field(description="Poll /status/{task_id} to track progress.")
    session_id: str = Field(
        description=(
            "Store this for followup calls. "
            "Matches the session_id in the request, or server-generated if absent."
        )
    )
    message: str = "Request queued successfully."


class TaskStatusResponse(BaseModel):
    """
    Returned by GET /status/{task_id}.
    The frontend polls this endpoint until status is 'complete' or 'failed'.

    status values:
      pending              — task created, background job not yet started
      running              — pipeline is executing
      complete             — result is populated; render the report
      failed               — error is populated; show error to user
    """
    task_id: str
    status: str = Field(description="pending | running | complete | failed")
    progress: Optional[str] = Field(
        default=None,
        description="Latest status message from the running pipeline node.",
    )
    thoughts: List[ThoughtSchema] = Field(
        default_factory=list,
        description=(
            "Accumulated reasoning steps from all completed nodes. "
            "Append new items to the frontend display as they arrive."
        ),
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full PipelineResult dict. Populated when status='complete'.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message. Populated when status='failed'.",
    )


class ClearSessionResponse(BaseModel):
    """Returned by DELETE /api/v1/analysis/{session_id}."""
    status: Literal["success", "already_cleared"]
    message: str
