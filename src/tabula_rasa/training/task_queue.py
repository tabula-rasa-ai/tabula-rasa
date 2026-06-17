"""Lightweight async task queue for background specialist training.

Uses Python stdlib threading + queue.Queue — no Redis dependency.
Supports:
- Enqueue training tasks from the API server
- Worker thread processes tasks in FIFO order
- Status tracking per task (queued, running, done, failed)
- Graceful shutdown

Usage:
    from tabula_rasa.training.task_queue import TaskQueue

    tq = TaskQueue(num_workers=1)

    @tq.register("train_specialist")
    def train_specialist_handler(task):
        op = task["op"]
        steps = task["steps"]
        run_training(op, steps)
        return {"op": op, "status": "done", "steps": steps}

    task_id = tq.enqueue("train_specialist", op="add", steps=5000)
    status = tq.get_status(task_id)  # {"state": "running", ...}
"""

import queue
import threading
import time
import uuid
from typing import Any, Callable, Optional


class TaskQueue:
    """Simple threaded task queue for background processing.

    Attributes:
        tasks: Dict of task_id -> task metadata.
        handlers: Dict of task_type -> handler function.
        _queue: Thread-safe queue.Queue.
        _workers: List of worker threads.
        _running: Whether the workers should keep running.
        _lock: Thread lock for task status updates.
    """

    def __init__(self, num_workers: int = 1) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.handlers: dict[str, Callable] = {}
        self._queue: queue.Queue = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._running = True
        self._lock = threading.Lock()

        for i in range(num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"task-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def register(self, task_type: str) -> Callable:
        """Decorator to register a handler for a task type.

        Args:
            task_type: String key matching the task type.

        Returns:
            Decorator function.
        """
        def decorator(fn: Callable) -> Callable:
            self.handlers[task_type] = fn
            return fn
        return decorator

    def enqueue(self, task_type: str, **kwargs: Any) -> str:
        """Enqueue a task for background processing.

        Args:
            task_type: Must match a registered handler.
            **kwargs: Arguments passed to the handler.

        Returns:
            task_id string for status tracking.

        Raises:
            ValueError: If no handler is registered for task_type.
        """
        if task_type not in self.handlers:
            raise ValueError(f"No handler registered for task type: {task_type}")

        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "type": task_type,
            "args": kwargs,
            "state": "queued",
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }

        with self._lock:
            self.tasks[task_id] = task
        self._queue.put(task_id)
        return task_id

    def get_status(self, task_id: str) -> Optional[dict[str, Any]]:
        """Get the status of a task by ID.

        Args:
            task_id: The task ID returned by enqueue().

        Returns:
            Task dict with state, result, error, or None if not found.
        """
        with self._lock:
            t = self.tasks.get(task_id)
            if t is None:
                return None
            return dict(t)  # Return a copy

    def list_tasks(self, state: Optional[str] = None,
                   limit: int = 50) -> list[dict[str, Any]]:
        """List tasks, optionally filtered by state.

        Args:
            state: Filter by state ('queued', 'running', 'done', 'failed').
            limit: Max tasks to return.

        Returns:
            List of task dicts (copies).
        """
        with self._lock:
            all_tasks = list(self.tasks.values())
        if state:
            all_tasks = [t for t in all_tasks if t["state"] == state]
        all_tasks.sort(key=lambda t: t["created_at"], reverse=True)
        return [dict(t) for t in all_tasks[:limit]]

    def _worker_loop(self) -> None:
        """Main worker loop — processes tasks from the queue."""
        while self._running:
            try:
                task_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task = None
            with self._lock:
                task = self.tasks.get(task_id)

            if task is None:
                continue

            handler = self.handlers.get(task["type"])
            if handler is None:
                with self._lock:
                    task["state"] = "failed"
                    task["error"] = f"No handler: {task['type']}"
                continue

            with self._lock:
                task["state"] = "running"
                task["started_at"] = time.time()

            try:
                result = handler(task["args"])
                with self._lock:
                    task["state"] = "done"
                    task["completed_at"] = time.time()
                    task["result"] = result
            except Exception as e:
                with self._lock:
                    task["state"] = "failed"
                    task["completed_at"] = time.time()
                    task["error"] = str(e)
            finally:
                self._queue.task_done()

    def shutdown(self, wait: bool = True, timeout: float = 30.0) -> None:
        """Shutdown the task queue and optionally wait for pending tasks.

        Args:
            wait: If True, wait for all queued tasks to complete.
            timeout: Max seconds to wait.
        """
        self._running = False
        if wait:
            self._queue.join()
        for w in self._workers:
            w.join(timeout=timeout)

    @property
    def stats(self) -> dict[str, Any]:
        """Get summary statistics about the task queue."""
        with self._lock:
            states = {}
            for t in self.tasks.values():
                s = t["state"]
                states[s] = states.get(s, 0) + 1
            return {
                "total": len(self.tasks),
                "queued": self._queue.qsize(),
                "states": states,
                "num_workers": len(self._workers),
            }
