"""Route traffic to experiment variants with session consistency."""
from __future__ import annotations

import random
from typing import Dict, Any


class TrafficRouter:
    """Deterministic router supporting weighted, geo and demographic routing."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._session_map: Dict[str, str] = {}

    def route(self, session_id: str, attributes: Dict[str, Any]) -> str:
        if session_id in self._session_map:
            return self._session_map[session_id]

        geo = attributes.get("geo")
        if geo and geo in self.cfg.geo_overrides:
            variant = self.cfg.geo_overrides[geo]
        else:
            demo = attributes.get("demo")
            if demo and demo in self.cfg.demo_overrides:
                variant = self.cfg.demo_overrides[demo]
            else:
                variant = self._weighted_choice()
        self._session_map[session_id] = variant
        return variant

    def _weighted_choice(self) -> str:
        rnd = random.random()
        cumulative = 0.0
        for variant, weight in self.cfg.traffic_split.items():
            cumulative += weight
            if rnd <= cumulative:
                return variant
        return variant  # pragma: no cover - fallback

    def update_split(self, new_split: Dict[str, float]) -> None:
        self.cfg.traffic_split = new_split
