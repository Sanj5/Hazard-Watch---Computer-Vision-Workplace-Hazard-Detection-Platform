from typing import List, Optional, Tuple

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

    vehicle_labels = {"forklift", "truck", "bus", "car", "robot", "motorcycle", "bicycle", "train"}
    dangerous_object_labels = {"knife", "scissors", "chainsaw", "hammer", "drill", "gas_cylinder"}

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def analyze(self, detections: List[Detection]) -> List[SceneHazardEvent]:
        events: List[SceneHazardEvent] = []

        persons = [d for d in detections if d.label == "person"]
        vehicles = [d for d in detections if d.label in self.vehicle_labels]
        dangerous_objects = [d for d in detections if d.label in self.dangerous_object_labels]

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
                closest: Optional[Tuple[Detection, float]] = None
                for v in vehicles:
                    vx, vy = self._center(v.bbox)
                    dist = ((px - vx) ** 2 + (py - vy) ** 2) ** 0.5
                    if closest is None or dist < closest[1]:
                        closest = (v, dist)

                if closest is not None and closest[1] < 140.0:
                    nearest_vehicle, min_dist = closest
                    events.append(
                        SceneHazardEvent(
                            hazard_type="worker_equipment_proximity",
                            confidence=max(0.5, min(1.0, 1.0 - min_dist / 140.0)),
                            severity=0.78,
                            details=(
                                f"Worker #{p.track_id} is close to {nearest_vehicle.label} "
                                f"({min_dist:.0f}px)"
                            ),
                        )
                    )

        # Alert when a dangerous object is near workers even without explicit collision prediction.
        if persons and dangerous_objects:
            for p in persons:
                px, py = self._center(p.bbox)
                nearest: Optional[Tuple[Detection, float]] = None
                for obj in dangerous_objects:
                    ox, oy = self._center(obj.bbox)
                    dist = ((px - ox) ** 2 + (py - oy) ** 2) ** 0.5
                    if nearest is None or dist < nearest[1]:
                        nearest = (obj, dist)

                if nearest is not None and nearest[1] < 160.0:
                    nearest_obj, min_dist = nearest
                    events.append(
                        SceneHazardEvent(
                            hazard_type="dangerous_object_proximity",
                            confidence=max(0.55, min(1.0, 1.0 - min_dist / 160.0)),
                            severity=0.82,
                            details=(
                                f"Worker #{p.track_id} is near {nearest_obj.label} "
                                f"({min_dist:.0f}px)"
                            ),
                        )
                    )

        return events
