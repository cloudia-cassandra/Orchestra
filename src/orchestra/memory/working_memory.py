"""Short-term working memory: a Redis-backed store shared by every agent working the same task.

Scoped entirely to one task_id (key prefix `orchestra:wm:{task_id}:`), so concurrent tasks never
collide. Holds the execution plan, every subtask attempt (intermediate results), the final
output per completed subtask, and an error log. Every key carries a TTL as a safety net —
`clear()` is the intended way a task's memory goes away (called once the task actually
completes), but a crash or a task that ends in `needs_escalation` (deliberately NOT cleared,
since a human still needs to look at it) shouldn't leak keys in Redis forever.
"""

import json
import os
from datetime import UTC, datetime
from typing import Any

from orchestra.orchestration.schemas import ExecutionPlan, SpecialistResult
from orchestra.memory import redis_client

DEFAULT_TTL_SECONDS = int(os.environ.get("ORCHESTRA_MEMORY_TTL_SECONDS", 60 * 60 * 24))


class WorkingMemory:
    def __init__(self, task_id: str, client=None, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.task_id = task_id
        self.client = client or redis_client.get_redis_client()
        self.ttl_seconds = ttl_seconds

    def _key(self, *parts: str) -> str:
        return ":".join(("orchestra", "wm", self.task_id, *parts))

    def _touch(self, key: str) -> None:
        self.client.expire(key, self.ttl_seconds)

    # ---------- task / plan ----------

    def set_task(self, task: str) -> None:
        key = self._key("task")
        self.client.set(key, task)
        self._touch(key)

    def get_task(self) -> str | None:
        return self.client.get(self._key("task"))

    def set_plan(self, plan: ExecutionPlan) -> None:
        key = self._key("plan")
        self.client.set(key, plan.model_dump_json())
        self._touch(key)

    def get_plan(self) -> ExecutionPlan | None:
        raw = self.client.get(self._key("plan"))
        return ExecutionPlan.model_validate_json(raw) if raw else None

    # ---------- subtask outputs ----------

    def add_completed_output(self, result: SpecialistResult) -> None:
        key = self._key("completed")
        self.client.hset(key, result.step_id, result.model_dump_json())
        self._touch(key)

    def get_completed_outputs(self) -> dict[str, SpecialistResult]:
        raw = self.client.hgetall(self._key("completed"))
        return {step_id: SpecialistResult.model_validate_json(value) for step_id, value in raw.items()}

    def add_intermediate_result(self, result: SpecialistResult) -> None:
        key = self._key("intermediate", result.step_id)
        self.client.rpush(key, result.model_dump_json())
        self._touch(key)

    def get_intermediate_results(self, step_id: str) -> list[SpecialistResult]:
        raw = self.client.lrange(self._key("intermediate", step_id), 0, -1)
        return [SpecialistResult.model_validate_json(item) for item in raw]

    # ---------- error log ----------

    def append_error_log(self, step_id: str | None, message: str, **extra: Any) -> None:
        key = self._key("errors")
        entry = {
            "step_id": step_id,
            "message": message,
            "timestamp": datetime.now(UTC).isoformat(),
            **extra,
        }
        self.client.rpush(key, json.dumps(entry))
        self._touch(key)

    def get_error_log(self) -> list[dict]:
        raw = self.client.lrange(self._key("errors"), 0, -1)
        return [json.loads(item) for item in raw]

    # ---------- lifecycle ----------

    def clear(self) -> None:
        keys = list(self.client.scan_iter(match=self._key("*")))
        if keys:
            self.client.delete(*keys)
