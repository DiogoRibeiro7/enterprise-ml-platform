"""Simple report scheduling abstraction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ReportScheduler:
    """Very small in-memory scheduler used for examples/tests."""

    jobs: list[tuple[str, Callable[[], None]]] = field(default_factory=list)

    def add_job(self, cron: str, job: Callable[[], None]) -> None:
        """Register a job.

        The ``cron`` expression is stored but not interpreted.  A production
        system could leverage APScheduler or a similar library to execute
        jobs based on this schedule.
        """
        self.jobs.append((cron, job))

    def run_all(self) -> None:
        """Execute all registered jobs immediately."""
        for _, job in self.jobs:
            job()
