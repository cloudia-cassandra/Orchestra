"""Generic tool registry: registration, domain authorization, rate limiting, invocation logging."""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from pydantic import BaseModel

from orchestra.orchestration.schemas import Domain

logger = logging.getLogger("orchestra.tools")


class RateLimitExceeded(RuntimeError):
    pass


class RateLimit(BaseModel):
    max_calls: int
    per_seconds: float


class ToolInvocationRecord(BaseModel):
    tool_name: str
    domain: Domain | None
    inputs: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
    success: bool
    latency_ms: float
    timestamp: datetime


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    allowed_domains: list[Domain]
    rate_limit: RateLimit
    handler: Callable[..., dict] = field(repr=False)


class ToolRegistry:
    """Holds tool definitions and mediates every invocation through one auditable path."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._call_history: dict[str, deque[float]] = defaultdict(deque)
        self.invocation_log: list[ToolInvocationRecord] = []

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name!r}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name!r}") from None

    def list_for_domain(self, domain: Domain) -> list[ToolSpec]:
        return [spec for spec in self._tools.values() if domain in spec.allowed_domains]

    def invoke(self, name: str, domain: Domain | None = None, **kwargs: Any) -> dict:
        spec = self.get(name)

        if domain is not None and domain not in spec.allowed_domains:
            raise PermissionError(
                f"Domain {domain!r} is not permitted to use tool {name!r} "
                f"(allowed: {spec.allowed_domains})"
            )

        self._enforce_rate_limit(spec)

        started = time.monotonic()
        timestamp = datetime.now(UTC)
        try:
            output = spec.handler(**kwargs)
        except Exception as exc:
            latency_ms = (time.monotonic() - started) * 1000
            self._log(
                ToolInvocationRecord(
                    tool_name=name,
                    domain=domain,
                    inputs=kwargs,
                    error=str(exc),
                    success=False,
                    latency_ms=latency_ms,
                    timestamp=timestamp,
                )
            )
            raise

        latency_ms = (time.monotonic() - started) * 1000
        self._log(
            ToolInvocationRecord(
                tool_name=name,
                domain=domain,
                inputs=kwargs,
                output=output,
                success=True,
                latency_ms=latency_ms,
                timestamp=timestamp,
            )
        )
        return output

    def _enforce_rate_limit(self, spec: ToolSpec) -> None:
        history = self._call_history[spec.name]
        now = time.monotonic()
        window_start = now - spec.rate_limit.per_seconds
        while history and history[0] < window_start:
            history.popleft()
        if len(history) >= spec.rate_limit.max_calls:
            raise RateLimitExceeded(
                f"Rate limit exceeded for tool {spec.name!r}: "
                f"{spec.rate_limit.max_calls} calls / {spec.rate_limit.per_seconds}s"
            )
        history.append(now)

    def _log(self, record: ToolInvocationRecord) -> None:
        self.invocation_log.append(record)
        level = logging.INFO if record.success else logging.WARNING
        logger.log(
            level,
            "tool=%s domain=%s success=%s latency_ms=%.1f",
            record.tool_name,
            record.domain,
            record.success,
            record.latency_ms,
        )
