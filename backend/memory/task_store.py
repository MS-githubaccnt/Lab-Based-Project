"""
memory/task_store.py
---------------------
TaskStore: abstract interface for task progress persistence.
InMemoryTaskStore: thread-safe in-process implementation (dev/testing).

The frontend polls get_task() to render live status messages, thoughts,
the final report, and any error details.

To use Redis in production, implement AbstractTaskStore and pass your
implementation to ChatService:
    chat_service = ChatService(task_store=RedisTaskStore(...))
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class AbstractTaskStore(ABC):
    """
    Interface all TaskStore implementations must satisfy.
    All methods are synchronous — wrap in run_in_executor if using async Redis.
    """

    @abstractmethod
    def initialize_task(self, task_id: str) -> None:
        """
        Create a new task record in 'pending' status.
        Called by ChatService.create_task() before the background job starts.
        """

    @abstractmethod
    def update_task_progress(
        self,
        task_id: str,
        status_message: str,
        new_thoughts: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Mark the task as 'running', update the status message,
        and append any new thoughts to the task's thought list.
        Called by the progress_callback after each pipeline node completes.
        """

    @abstractmethod
    def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Mark the task as complete and store the full PipelineResult dict.
        """

    @abstractmethod
    def fail_task(self, task_id: str, error_message: str) -> None:
        """
        Mark the task as 'failed' and store the error message.
        """

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the current task state dict, or None if not found.
        """


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryTaskStore(AbstractTaskStore):
    """
    Thread-safe in-process task store backed by a plain dict.
    Suitable for development, testing, and single-process deployments.

    For production with multiple workers or process restarts,
    replace with a Redis-backed implementation.

    Task state schema:
        task_id                 str
        status                  'pending' | 'running' | 'complete' | 'failed'
        progress                str     — latest status message
        thoughts                list    — all Thought dicts emitted so far
        result                  dict | None   — PipelineResult on completion
        error                   str | None    — when status=failed
        created_at              float   — unix timestamp
        updated_at              float   — unix timestamp
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock  = threading.Lock()

    def initialize_task(self, task_id: str) -> None:
        with self._lock:
            self._store[task_id] = {
                "task_id":               task_id,
                "status":                "pending",
                "progress":              "Initialising...",
                "thoughts":              [],
                "result":                None,
                "error":                 None,
                "created_at":            time.time(),
                "updated_at":            time.time(),
            }

    def update_task_progress(
        self,
        task_id: str,
        status_message: str,
        new_thoughts: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        with self._lock:
            task = self._store.get(task_id)
            if task is None:
                return  # Task was cleared or never created — silently drop
            task["status"]     = "running"
            task["progress"]   = status_message
            task["updated_at"] = time.time()
            if new_thoughts:
                task["thoughts"].extend(new_thoughts)

    def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any],
    ) -> None:
        with self._lock:
            task = self._store.get(task_id)
            if task is None:
                return

            pipeline_status = result.get("status", "complete")

            if pipeline_status == "followup_answered":
                task["status"]   = "complete"
                task["progress"] = "Follow-up answered."
            else:
                task["status"]   = "complete"
                task["progress"] = "Analysis complete."

            task["result"]     = result
            task["updated_at"] = time.time()

            # Append any thoughts included in the final result
            final_thoughts = result.get("thoughts") or []
            if final_thoughts:
                # Avoid duplicating thoughts already added via update_task_progress
                existing_count = len(task["thoughts"])
                if len(final_thoughts) > existing_count:
                    task["thoughts"] = final_thoughts

    def fail_task(self, task_id: str, error_message: str) -> None:
        with self._lock:
            task = self._store.get(task_id)
            if task is None:
                return
            task["status"]     = "failed"
            task["progress"]   = "Analysis failed."
            task["error"]      = error_message
            task["updated_at"] = time.time()

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self._store.get(task_id)
            if task is None:
                return None
            return deepcopy(task)  # return a copy so callers can't mutate store state

    def delete_task(self, task_id: str) -> None:
        """Remove a task from the store (used by clear_session)."""
        with self._lock:
            self._store.pop(task_id, None)
