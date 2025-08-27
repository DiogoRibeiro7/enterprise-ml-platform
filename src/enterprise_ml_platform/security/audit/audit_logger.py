"""Audit logging utilities."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class AuditLogger:
    """Small append‑only audit logger.

    Events are written as JSON lines with a hash of the previous line so
    that tampering can be detected during verification.  This does not
    replace a fully fledged log management solution but is sufficient for
    tests and local development.
    """

    log_file: Path
    _last_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.log_file.exists():
            try:\n                *_, last = self.log_file.read_text().splitlines()
                self._last_hash = json.loads(last)["hash"]
            except Exception:  # pragma: no cover - best effort
                self._last_hash = ""

    def log(self, event: str, **payload: Any) -> None:
        """Log an ``event`` with optional ``payload`` data."""

        entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "payload": payload,
            "prev_hash": self._last_hash,
        }
        data = json.dumps(entry, sort_keys=True).encode("utf-8")
        entry_hash = hashlib.sha256(data).hexdigest()
        entry["hash"] = entry_hash
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        self._last_hash = entry_hash

