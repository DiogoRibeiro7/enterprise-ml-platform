"""Utilities for managing semantic versions of registered models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VersionManager:
    """Maintain semantic versions for models.

    The manager keeps an in-memory mapping from model name to a list of
    released versions.  Versions follow the ``MAJOR.MINOR.PATCH`` scheme.  The
    :meth:`next_version` helper increments the patch component by default but
    allows explicit bumps of any segment.
    """

    versions: dict[str, list[str]] = field(default_factory=dict)

    def next_version(
        self,
        model: str,
        bump: tuple[int, int, int] | None = None,
    ) -> str:
        """Return the next semantic version for ``model``.

        Args:
            model: Model identifier.
            bump: Optional tuple specifying how much to bump the
                ``(major, minor, patch)`` components of the latest version.
                If not provided the patch component is incremented by one.
        """

        history = self.versions.setdefault(model, [])
        if not history:
            version = "1.0.0"
        else:
            major, minor, patch = map(int, history[-1].split("."))
            if bump:
                major += bump[0]
                minor += bump[1]
                patch += bump[2]
            else:
                patch += 1
            version = f"{major}.{minor}.{patch}"
        history.append(version)
        return version
