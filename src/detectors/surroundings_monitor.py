from typing import List

from src.utils.types import Detection, SceneHazardEvent


class SurroundingsMonitor:
    """Detect scene-level hazards from detected objects and their spatial context."""

    explicit_hazard_labels = {
        "fire": 1.0,
        "smoke": 0.95,
        "spill": 0.85,
        "wet_floor": 0.8,
        "open_pit": 0.95,
        "exposed_wires": 0.9,
    }

    vehicle_labels = {"forklift", "truck", "bus", "car", "robot"}

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def analyze(self, detections: List[Detection]) -> List[SceneHazardEvent]:
        events: List[SceneHazardEvent] = []

        persons = [d for d in detections if d.label == "person"]
        vehicles = [d for d in detections if d.label in self.vehicle_labels]

        # Direct hazards detected in the scene regardless of worker behavior.
        for d in detections:
            if d.label in self.explicit_hazard_labels:
                sev = self.explicit_hazard_labels[d.label]
                events.append(
                    SceneHazardEvent(
                        hazard_type=d.label,
                        confidence=d.confidence,
                        severity=sev,
                        details=f"Detected environmental hazard: {d.label}",
                    )
                )

        # Congested equipment zone can be dangerous even before a near-miss.
        if len(vehicles) >= 3:
            events.append(
                SceneHazardEvent(
                    hazard_type="equipment_congestion",
                    confidence=min(1.0, 0.55 + 0.1 * len(vehicles)),
                    severity=0.75,
                    details=f"High equipment density: {len(vehicles)} vehicles in frame",
                )
            )

        # Worker proximity to heavy equipment as surroundings risk.
        if persons and vehicles:
            for p in persons:
                px, py = self._center(p.bbox)
                closest = None
                for v in vehicles:
                    vx, vy = self._center(v.bbox)
                    dist = ((px - vx) ** 2 + (py - vy) ** 2) ** 0.5
                    closest = dist if closest is None else min(closest, dist)

                if closest is not None and closest < 120.0:
                    events.append(
                        SceneHazardEvent(
                            hazard_type="worker_equipment_proximity",
                            confidence=max(0.5, min(1.0, 1.0 - closest / 120.0)),
                            severity=0.7,
                            details=f"Worker #{p.track_id} very close to equipment",
                        )
                    )

        return events
