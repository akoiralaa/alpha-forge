"""Health check system — monitors all subsystem liveness.

Each component registers a health check function. The aggregator
runs all checks and reports overall system health.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


@dataclass
class ComponentHealth:
    name: str
    status: HealthStatus
    message: str = ""
    last_check_ns: int = 0
    latency_ms: float = 0.0


@dataclass
class SystemHealth:
    status: HealthStatus
    components: list[ComponentHealth] = field(default_factory=list)
    timestamp_ns: int = 0

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def unhealthy_components(self) -> list[ComponentHealth]:
        return [c for c in self.components if c.status == HealthStatus.UNHEALTHY]

    @property
    def degraded_components(self) -> list[ComponentHealth]:
        return [c for c in self.components if c.status == HealthStatus.DEGRADED]


# Type for health check callables
HealthCheckFn = Callable[[], tuple[HealthStatus, str]]


class HealthChecker:
    """Aggregates health checks from all subsystems."""

    def __init__(self):
        self._checks: dict[str, HealthCheckFn] = {}
        self._history: list[SystemHealth] = []

    def register(self, name: str, check_fn: HealthCheckFn):
        """Register a health check for a named component."""
        self._checks[name] = check_fn

    def unregister(self, name: str):
        self._checks.pop(name, None)

    def check(self) -> SystemHealth:
        """Run all health checks and return aggregated status."""
        components: list[ComponentHealth] = []
        now = time.time_ns()

        for name, fn in self._checks.items():
            start = time.time_ns()
            try:
                status, message = fn()
            except Exception as e:
                status = HealthStatus.UNHEALTHY
                message = f"check raised: {e}"
            elapsed_ms = (time.time_ns() - start) / 1e6

            components.append(ComponentHealth(
                name=name,
                status=status,
                message=message,
                last_check_ns=now,
                latency_ms=elapsed_ms,
            ))

        # Aggregate: worst status wins
        if any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        result = SystemHealth(
            status=overall,
            components=components,
            timestamp_ns=now,
        )
        self._history.append(result)
        return result

    @property
    def last_check(self) -> Optional[SystemHealth]:
        return self._history[-1] if self._history else None

    @property
    def registered_components(self) -> list[str]:
        return list(self._checks.keys())
