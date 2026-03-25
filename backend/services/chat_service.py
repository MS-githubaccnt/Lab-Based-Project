"""
chat_service.py
----------------
ChatService: the service layer between your API routes and the
eco-material selection pipeline.

Responsibilities
----------------
  create_task()
      Generate a task ID and register it in the task store so the
      frontend can begin polling immediately, before the background
      job starts.

  get_task_status(task_id)
      Return the current task state dict for the frontend to render.

  start_analysis_background(...)
      Begin a new eco-material analysis run. Called as a background task
      from your API route so the HTTP response returns immediately.

  resume_clarification_background(...)
      Resume a pipeline that paused for clarification. Called as a
      background task after the user submits their answer.

  ask_followup_background(...)
      Answer a follow-up question about the completed recommendation.
      No pipeline re-run — handled by ExplanationNode directly.

  clear_session(session_id)
      Invalidate a LangGraph session so it cannot be resumed or followed up.
      Does not delete task history — only blocks future pipeline operations
      on this session.

Usage (FastAPI example)
-----------------------
    chat_service = ChatService()

    @app.post("/analysis/start")
    async def start(req: StartRequest, background_tasks: BackgroundTasks):
        task_id = chat_service.create_task()
        session_id = req.session_id or str(uuid.uuid4())
        background_tasks.add_task(
            chat_service.start_analysis_background,
            task_id=task_id,
            session_id=session_id,
            cad_file_path=req.cad_file_path,
            object_description=req.object_description,
        )
        return {"task_id": task_id, "session_id": session_id}

    @app.get("/analysis/status/{task_id}")
    def status(task_id: str):
        return chat_service.get_task_status(task_id)

    @app.post("/analysis/clarify")
    async def clarify(req: ClarifyRequest, background_tasks: BackgroundTasks):
        task_id = chat_service.create_task()
        background_tasks.add_task(
            chat_service.resume_clarification_background,
            task_id=task_id,
            session_id=req.session_id,
            clarification_answer=req.answer,
        )
        return {"task_id": task_id}

    @app.post("/analysis/followup")
    async def followup(req: FollowupRequest, background_tasks: BackgroundTasks):
        task_id = chat_service.create_task()
        background_tasks.add_task(
            chat_service.ask_followup_background,
            task_id=task_id,
            session_id=req.session_id,
            question=req.question,
        )
        return {"task_id": task_id}
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from memory.task_store import AbstractTaskStore, InMemoryTaskStore
from core.orchestrator import EcoMaterialOrchestrator
from schema.state import PipelineResult, Thought


class ChatService:
    """
    Service layer between API routes and the eco-material pipeline.

    Args:
        task_store:
            Persistence layer for task progress. Defaults to InMemoryTaskStore.
            Pass a Redis-backed implementation for production deployments.
        orchestrator:
            The pipeline orchestrator. Defaults to EcoMaterialOrchestrator().
            Override in tests to inject a mock.
    """

    def __init__(
        self,
        task_store: Optional[AbstractTaskStore] = None,
        orchestrator: Optional[EcoMaterialOrchestrator] = None,
    ) -> None:
        self.task_store  = task_store  or InMemoryTaskStore()
        self.orchestrator = orchestrator or EcoMaterialOrchestrator()

        # Tracks session IDs that have been explicitly cleared.
        # Operations on cleared sessions are rejected with a clear error.
        self._cleared_sessions: set[str] = set()

    # =======================================================================
    # Task management
    # =======================================================================

    def create_task(self) -> str:
        """
        Generate a unique task ID and register it in the task store.

        Call this BEFORE spawning the background task so the frontend
        can begin polling immediately without a race condition.

        Returns:
            task_id: UUID string the frontend uses for polling.
        """
        task_id = str(uuid.uuid4())
        self.task_store.initialize_task(task_id)
        return task_id

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Return the current task state for the frontend to render.

        Returns a dict with keys:
            task_id, status, progress, thoughts, result,
            clarification_question, error, created_at, updated_at.

        Returns {"error": "Task not found"} if the task_id is unknown.
        """
        task = self.task_store.get_task(task_id)
        if task is None:
            return {"error": f"Task '{task_id}' not found."}
        return task

    # =======================================================================
    # Background operations
    # =======================================================================

    async def start_analysis_background(
        self,
        task_id: str,
        session_id: str,
        cad_file_path: str,
        object_description: str,
        recyclability_priority: float = 0.7,
    ) -> None:
        """
        Run a new eco-material analysis pipeline in the background.

        Called as a background task from your API route. The HTTP response
        returns before this completes. The frontend polls get_task_status()
        to track progress.

        Args:
            task_id:                From create_task() — already registered.
            session_id:             Unique identifier for this analysis session.
                                    Required for follow-up and resume calls.
            cad_file_path:          Absolute path to the .step, .stp, or .stl file.
            object_description:     Plain-language part description.
            recyclability_priority: Eco weighting 0.0–1.0. Default 0.7.
        """
        if session_id in self._cleared_sessions:
            self.task_store.fail_task(
                task_id,
                f"Session '{session_id}' has been cleared. "
                "Start a new session to run a fresh analysis."
            )
            return

        self.task_store.update_task_progress(task_id, "Starting analysis...")

        try:
            result = await self.orchestrator.process_request(
                cad_file_path=cad_file_path,
                object_description=object_description,
                session_id=session_id,
                recyclability_priority=recyclability_priority,
                progress_callback=self._make_progress_callback(task_id),
            )
            self.task_store.complete_task(task_id, result.to_dict())

        except Exception as exc:
            self.task_store.fail_task(task_id, str(exc))
            raise

    async def resume_clarification_background(
        self,
        task_id: str,
        session_id: str,
        clarification_answer: str,
    ) -> None:
        """
        Resume a pipeline that paused to ask a clarification question.

        Call this after the user answers the question shown in a task
        with status='clarification_needed'.

        Args:
            task_id:                From a new create_task() call.
            session_id:             The session_id from the original start call.
            clarification_answer:   User's answer to the clarification question.
        """
        self._check_session(task_id, session_id)
        self.task_store.update_task_progress(
            task_id, "Resuming analysis with your answer..."
        )

        try:
            result = await self.orchestrator.resume_with_clarification(
                session_id=session_id,
                clarification_answer=clarification_answer,
                progress_callback=self._make_progress_callback(task_id),
            )
            self.task_store.complete_task(task_id, result.to_dict())

        except Exception as exc:
            self.task_store.fail_task(task_id, str(exc))
            raise

    async def ask_followup_background(
        self,
        task_id: str,
        session_id: str,
        question: str,
    ) -> None:
        """
        Answer a follow-up question about the completed recommendation.

        No pipeline re-run. ExplanationNode handles the question directly
        using the pipeline context stored in the LangGraph checkpoint.

        Args:
            task_id:    From a new create_task() call.
            session_id: The session_id from the original completed run.
            question:   The engineer's follow-up question.
        """
        self._check_session(task_id, session_id)
        self.task_store.update_task_progress(
            task_id, "Answering your follow-up question..."
        )

        try:
            result = await self.orchestrator.ask_followup(
                session_id=session_id,
                question=question,
                progress_callback=self._make_progress_callback(task_id),
            )
            self.task_store.complete_task(task_id, result.to_dict())

        except Exception as exc:
            self.task_store.fail_task(task_id, str(exc))
            raise

    # =======================================================================
    # Session management
    # =======================================================================

    def clear_session(self, session_id: str) -> bool:
        """
        Invalidate a session so it cannot be resumed or followed up.

        This does NOT delete task history — existing task_ids for this
        session remain readable via get_task_status(). It only prevents
        future pipeline operations on this session_id.

        To start fresh with the same session_id, call start_analysis_background()
        again — the LangGraph checkpointer will overwrite the old state.

        Returns:
            True if the session was active and is now cleared.
            False if the session was already cleared or never started.
        """
        try:
            if session_id in self._cleared_sessions:
                return False
            self._cleared_sessions.add(session_id)
            return True
        except Exception:
            return False

    # =======================================================================
    # Internal helpers
    # =======================================================================

    def _check_session(self, task_id: str, session_id: str) -> None:
        """
        Raise and fail the task if the session has been cleared.
        """
        if session_id in self._cleared_sessions:
            msg = (
                f"Session '{session_id}' has been cleared and cannot be resumed. "
                "Start a new analysis with start_analysis_background()."
            )
            self.task_store.fail_task(task_id, msg)
            raise ValueError(msg)

    def _make_progress_callback(self, task_id: str):
        """
        Build an async progress callback bound to a specific task_id.

        The callback signature matches what the orchestrator expects:
            async fn(node_name: str, status_message: str, new_thoughts: List[Thought])

        Fires task_store.update_task_progress() after each node completes,
        pushing the latest status and thoughts so the frontend can render
        real-time reasoning steps.
        """
        task_store = self.task_store

        async def _callback(
            node_name: str,
            status_message: str,
            new_thoughts: List[Thought],
        ) -> None:
            # Convert Thought TypedDicts to plain dicts for serialisation
            serialised_thoughts = [dict(t) for t in (new_thoughts or [])]
            task_store.update_task_progress(
                task_id,
                status_message,
                new_thoughts=serialised_thoughts,
            )

        return _callback