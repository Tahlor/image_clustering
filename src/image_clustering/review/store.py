"""Durable, atomic persistence for reviewer decisions.

Decisions live in their own ``review_labels`` namespace so they never overwrite
canonical clustering, cropping, or evaluation report artifacts. Every save is
written to a temporary file and replaced into place, so an interrupted write
cannot truncate a reviewer's work.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

from image_clustering.review.decisions import empty_state, utc_now

LOGGER = logging.getLogger(__name__)


class DecisionStore:
    """Load and atomically persist the reviewer decision document."""

    def __init__(self, path: Path, provenance: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self._provenance = dict(provenance or {})
        self._lock = threading.Lock()
        self._state = self._load()

    @classmethod
    def for_output_root(
        cls,
        output_root: Path,
        provenance: dict[str, Any] | None = None,
    ) -> DecisionStore:
        """Return the store for the canonical decisions file of a run."""
        path = Path(output_root) / "review_labels" / "decisions.json"
        return cls(path, provenance=provenance)

    @property
    def state(self) -> dict[str, Any]:
        """Return the mutable in-memory decision document."""
        return self._state

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return empty_state(self._provenance)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning(
                "Could not read %s (%s); starting a new document", self.path, error
            )
            self._quarantine()
            return empty_state(self._provenance)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("clusters"), dict
        ):
            LOGGER.warning("Ignoring unusable decision document at %s", self.path)
            self._quarantine()
            return empty_state(self._provenance)
        if self._provenance:
            payload["provenance"] = {
                **payload.get("provenance", {}),
                **self._provenance,
            }
        return payload

    def _quarantine(self) -> None:
        """Preserve an unreadable decision file instead of silently discarding it."""
        if not self.path.is_file():
            return
        backup = self.path.with_name(f"{self.path.stem}.unreadable{self.path.suffix}")
        try:
            shutil.copy2(self.path, backup)
            LOGGER.warning("Preserved the previous decision file at %s", backup)
        except OSError as error:  # pragma: no cover - defensive
            LOGGER.warning("Could not preserve %s (%s)", self.path, error)

    def save(self) -> Path:
        """Persist the current document atomically and return its path."""
        with self._lock:
            self._state["saved_at"] = utc_now()
            if self._provenance:
                self._state["provenance"] = {
                    **self._state.get("provenance", {}),
                    **self._provenance,
                }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self._state, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        return self.path

    def replace(self, decisions: dict[str, Any]) -> Path:
        """Replace all cluster decisions, for example from an imported file."""
        if not isinstance(decisions, dict):
            raise ValueError("decisions must be an object keyed by cluster id")
        self._state["clusters"] = decisions
        return self.save()
