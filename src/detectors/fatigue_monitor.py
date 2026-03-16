from typing import Dict, List, Tuple
import math

from src.utils.types import BehaviorEvent, Detection, FatigueEvent


class FatigueMonitor:
    """Heuristic fatigue monitor based on prolonged low movement and posture cues."""

    def __init__(self, movement_threshold_px_s: float = 6.0, stationary_alert_seconds: float = 30.0, min_score_for_alert: float = 0.65):
        self.movement_threshold_px_s = movement_threshold_px_s
        self.stationary_alert_seconds = stationary_alert_seconds
        self.min_score_for_alert = min_score_for_alert
        self._last_center: Dict[int, Tuple[float, float]] = {}
        self._stationary_seconds: Dict[int, float] = {}

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def analyze(self, detections: List[Detection], behavior_events: List[BehaviorEvent], dt_seconds: float) -> List[FatigueEvent]:
        fatigue_events: List[FatigueEvent] = []
        posture_boost = 0.0
        if any(e.event_type == "unsafe_posture" and e.confidence >= 0.6 for e in behavior_events):
            posture_boost = 0.2

        persons = [d for d in detections if d.label == "person"]
        for p in persons:
            c = self._center(p.bbox)
            prev = self._last_center.get(p.track_id)
            speed = 0.0
            if prev is not None and dt_seconds > 1e-6:
                speed = math.dist(c, prev) / dt_seconds

            stationary = self._stationary_seconds.get(p.track_id, 0.0)
            if speed < self.movement_threshold_px_s:
                stationary += dt_seconds
            else:
                stationary = max(0.0, stationary - 0.5 * dt_seconds)

            self._last_center[p.track_id] = c
            self._stationary_seconds[p.track_id] = stationary

            base = min(1.0, stationary / max(1.0, self.stationary_alert_seconds))
            score = max(0.0, min(1.0, base + posture_boost))
            if score >= self.min_score_for_alert:
                fatigue_events.append(
                    FatigueEvent(
                        track_id=p.track_id,
                        fatigue_score=score,
                        stationary_seconds=stationary,
                        confidence=max(0.5, min(1.0, p.confidence)),
                    )
                )

        return fatigue_events
