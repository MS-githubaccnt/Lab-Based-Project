"""
controllers/analysis_controller.py
------------------------------------
FastAPI router for the eco-material analysis pipeline.

Routes
------
  POST   /api/v1/analysis/start
      Queue a new eco-material analysis. Returns task_id immediately.
      Frontend polls /status/{task_id} to track progress.

  POST   /api/v1/analysis/clarify
      Submit an answer to a clarification question.
      Call after polling returns status='clarification_needed'.

  POST   /api/v1/analysis/followup
      Ask a follow-up question about a completed recommendation.
      No pipeline re-run — ExplanationNode answers directly from context.

  GET    /api/v1/analysis/status/{task_id}
      Poll task progress. Returns status, thoughts, result, or error.

  DELETE /api/v1/analysis/{session_id}
      Invalidate a session. Blocks future clarify/followup on this session.

Dependency injection
--------------------
ChatService is a MODULE-LEVEL SINGLETON — not instantiated per-request.

This is critical: our orchestrator uses LangGraph's MemorySaver to store
pipeline checkpoints between process_request() and resume_with_clarification()
/ ask_followup() calls. If ChatService were re-created per-request (like the
reference pattern), each instance would have its own empty MemorySaver and
resume/followup calls would find no checkpoint.

To use a Redis-backed task store or checkpointer in production, instantiate
the service once at startup and pass it in:
    _chat_service = ChatService(task_store=RedisTaskStore(...))
"""

from __future__ import annotations

import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
)
from pathlib import Path
from schema.api_models import (
    AcceptedResponse,
    ClarifyRequest,
    ClearSessionResponse,
    ContinueAnalysisRequest,
    FollowupRequest,
    StartAnalysisRequest,
    TaskStatusResponse,
    ThoughtSchema,
)
from services.chat_service import ChatService
from tools.cad_parser_tools import parse_and_extract_cad_features


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".stl", ".step", ".stp"}
MAX_FILE_SIZE_MB = 50  # optional safeguard
# ---------------------------------------------------------------------------
# Singleton service
# ---------------------------------------------------------------------------

# ChatService holds the LangGraph MemorySaver — must be a single instance
# shared across all requests so checkpoints persist between calls.
_chat_service = ChatService()


def get_chat_service() -> ChatService:
    """
    FastAPI dependency that returns the singleton ChatService.
    Injected via Depends() into every route handler.
    """
    return _chat_service


async def _save_uploaded_cad(file: UploadFile, allowed_extensions: set[str]) -> tuple[str, Path]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name.")

    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}",
        )

    upload_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{upload_id}{ext}"

    try:
        size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",
                    )

                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    finally:
        await file.close()

    return upload_id, file_path


def _cad_path_from_upload_id(upload_id: str) -> Path:
    safe_upload_id = Path(upload_id).name
    for ext in ALLOWED_EXTENSIONS:
        candidate = UPLOAD_DIR / f"{safe_upload_id}{ext}"
        if candidate.exists():
            return candidate
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Uploaded CAD file '{upload_id}' was not found.",
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/analysis",
    tags=["Eco-Material Analysis"],
)


# =============================================================================
# POST /parse — upload an STL and run only the CAD parser
# =============================================================================

@router.post(
    "/parse",
    summary="Upload STL and preview CAD parser features",
)
async def parse_cad_file(
    file: UploadFile = File(...),
) -> dict:
    upload_id, file_path = await _save_uploaded_cad(file, {".stl"})

    try:
        geometry = parse_and_extract_cad_features.invoke({
            "file_path": str(file_path),
        })
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CAD parser failed: {str(e)}",
        )

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "geometry": geometry,
    }


# =============================================================================
# POST /continue — start full analysis for a previously parsed upload
# =============================================================================

@router.post(
    "/continue",
    response_model=AcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Continue material analysis after CAD parser preview",
)
async def continue_analysis(
    payload: ContinueAnalysisRequest,
    background_tasks: BackgroundTasks,
    service: ChatService = Depends(get_chat_service),
) -> AcceptedResponse:
    file_path = _cad_path_from_upload_id(payload.upload_id)
    session_id = str(uuid.uuid4())
    task_id = service.create_task()

    background_tasks.add_task(
        service.start_analysis_background,
        task_id=task_id,
        session_id=session_id,
        cad_file_path=str(file_path),
        object_description=payload.object_description,
        recyclability_priority=payload.recyclability_priority,
    )

    return AcceptedResponse(
        task_id=task_id,
        session_id=session_id,
        message=(
            f"Analysis started for '{payload.object_description}'. "
            f"Poll /status/{task_id} for progress."
        ),
    )


# =============================================================================
# POST /start — begin a new analysis
# =============================================================================

@router.post(
    "/start",
    response_model=AcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start eco-material analysis (with CAD upload)",
)
async def start_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    object_description: str = Form(...),
    recyclability_priority: float = Form(0.7),
    service: ChatService = Depends(get_chat_service),
) -> AcceptedResponse:
    # -------------------------
    # Generate IDs
    # -------------------------
    session_id = str(uuid.uuid4())
    task_id = service.create_task()

    # -------------------------
    # Save file safely
    # -------------------------
    _, file_path = await _save_uploaded_cad(file, ALLOWED_EXTENSIONS)

    # -------------------------
    # Start background pipeline
    # -------------------------
    background_tasks.add_task(
        service.start_analysis_background,
        task_id=task_id,
        session_id=session_id,
        cad_file_path=str(file_path),
        object_description=object_description,
        recyclability_priority=recyclability_priority,
    )

    # -------------------------
    # Immediate response
    # -------------------------
    return AcceptedResponse(
        task_id=task_id,
        session_id=session_id,
        message=(
            f"Analysis started for '{object_description}'. "
            f"Poll /status/{task_id} for progress."
        ),
    )

# =============================================================================
# POST /clarify — submit answer to clarification question
# =============================================================================

@router.post(
    "/clarify",
    response_model=AcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit clarification answer",
    description=(
        "Called after polling returns status='clarification_needed'. "
        "Submit the user's answer to the clarification question shown in the "
        "task status. The pipeline resumes in the background."
    ),
)
async def submit_clarification(
    payload: ClarifyRequest,
    background_tasks: BackgroundTasks,
    service: ChatService = Depends(get_chat_service),
) -> AcceptedResponse:
    # Validate that the session exists and has not been cleared.
    # A cleared session raises ValueError in the background method —
    # we surface it here as 409 Conflict before queuing the task.
    if payload.session_id in service._cleared_sessions:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Session '{payload.session_id}' has been cleared and cannot "
                "be resumed. Start a new analysis with POST /start."
            ),
        )

    task_id = service.create_task()

    background_tasks.add_task(
        service.resume_clarification_background,
        task_id=task_id,
        session_id=payload.session_id,
        clarification_answer=payload.clarification_answer,
    )

    return AcceptedResponse(
        task_id=task_id,
        session_id=payload.session_id,
        message=(
            f"Clarification received. Resuming analysis. "
            f"Poll /status/{task_id} for progress."
        ),
    )


# =============================================================================
# POST /followup — ask a follow-up question
# =============================================================================

@router.post(
    "/followup",
    response_model=AcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ask a follow-up question",
    description=(
        "Ask a follow-up question about a completed recommendation. "
        "The pipeline does not re-run — ExplanationNode answers directly "
        "using the full pipeline context stored in the session. "
        "Requires a session that has previously reached status='complete'."
    ),
)
async def ask_followup(
    payload: FollowupRequest,
    background_tasks: BackgroundTasks,
    service: ChatService = Depends(get_chat_service),
) -> AcceptedResponse:
    if payload.session_id in service._cleared_sessions:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Session '{payload.session_id}' has been cleared. "
                "Start a new analysis with POST /start."
            ),
        )

    task_id = service.create_task()

    background_tasks.add_task(
        service.ask_followup_background,
        task_id=task_id,
        session_id=payload.session_id,
        question=payload.question,
    )

    return AcceptedResponse(
        task_id=task_id,
        session_id=payload.session_id,
        message=(
            f"Follow-up question queued. "
            f"Poll /status/{task_id} for the answer."
        ),
    )


# =============================================================================
# GET /status/{task_id} — poll task progress
# =============================================================================

@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    summary="Poll task progress",
    description=(
        "Poll this endpoint after any POST route. "
        "Returns the current pipeline status, live reasoning thoughts, "
        "and the final result or error when complete. "
        "Recommended polling interval: 1–2 seconds."
    ),
)
async def get_task_status(
    task_id: str,
    service: ChatService = Depends(get_chat_service),
) -> TaskStatusResponse:
    task_data = service.get_task_status(task_id)

    # get_task_status returns {"error": "Task '...' not found."} for unknown IDs
    if "error" in task_data and task_data.get("task_id") is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )

    # Coerce thoughts to ThoughtSchema — use .get() with defaults throughout
    # so a task that was initialised but not yet updated never raises KeyError
    raw_thoughts = task_data.get("thoughts") or []
    thoughts = [
        ThoughtSchema(
            node=t.get("node", "unknown"),
            type=t.get("type", "info"),
            text=t.get("text", ""),
        )
        for t in raw_thoughts
    ]

    return TaskStatusResponse(
        task_id=task_id,
        status=task_data.get("status", "pending"),
        progress=task_data.get("progress"),
        thoughts=thoughts,
        result=task_data.get("result"),
        clarification_question=task_data.get("clarification_question"),
        error=task_data.get("error"),
    )


# =============================================================================
# DELETE /{session_id} — clear session
# =============================================================================

@router.delete(
    "/{session_id}",
    response_model=ClearSessionResponse,
    summary="Clear analysis session",
    description=(
        "Invalidate a session so it cannot be resumed or followed up. "
        "Call this when the user starts a new part or explicitly clears the session. "
        "Task history (for already-completed tasks) remains readable via /status. "
        "To re-analyse the same part, start a new session with POST /start."
    ),
)
async def clear_session(
    session_id: str,
    service: ChatService = Depends(get_chat_service),
) -> ClearSessionResponse:
    cleared = service.clear_session(session_id)

    if not cleared:
        # Session was already cleared — not an error, just idempotent
        return ClearSessionResponse(
            status="already_cleared",
            message=(
                f"Session '{session_id}' was already cleared. "
                "No action taken."
            ),
        )

    return ClearSessionResponse(
        status="success",
        message=(
            f"Session '{session_id}' cleared. "
            "Future clarify and follow-up calls on this session will be rejected."
        ),
    )
