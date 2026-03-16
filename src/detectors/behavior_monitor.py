from typing import List

from src.utils.types import Detection, BehaviorEvent


class BehaviorMonitor:
    """
    Maps behavior class detections into normalized unsafe behavior events.
    Example labels your detector can emit: running, phone_use, climbing_guardrail.
    """

    severity_map = {
        "unsafe_posture": 0.65,
        "running": 0.4,
        "phone_use": 0.5,
        "climbing_guardrail": 0.9,
        "no_harness": 0.95,
        "fallen_person": 1.0,
    }

    def analyze(self, detections: List[Detection], extra_events: List[BehaviorEvent] | None = None) -> List[BehaviorEvent]:
        events: List[BehaviorEvent] = []
        for d in detections:
            if d.label in self.severity_map:
                events.append(
                    BehaviorEvent(
                        event_type=d.label,
                        track_id=d.track_id,
                        confidence=d.confidence,
                        severity=self.severity_map[d.label],
                    )
                )
        if extra_events:
            events.extend(extra_events)
        return events
