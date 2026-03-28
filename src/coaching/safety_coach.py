from typing import List

from src.utils.types import PPEViolation, BehaviorEvent, NearMissEvent, CollisionRiskEvent, FatigueEvent, SceneHazardEvent


class SafetyCoach:
    def generate(
        self,
        ppe_violations: List[PPEViolation],
        behavior_events: List[BehaviorEvent],
        near_miss_events: List[NearMissEvent],
        collision_events: List[CollisionRiskEvent],
        fatigue_events: List[FatigueEvent],
        scene_hazard_events: List[SceneHazardEvent],
        risk_level: str,
    ) -> List[str]:
        messages: List[str] = []

        # Prioritize immediate injury-risk alerts so they survive top-3 truncation.
        if collision_events:
            soonest = min(collision_events, key=lambda c: c.predicted_seconds)
            if soonest.predicted_seconds <= 1.5:
                messages.append(
                    f"Immediate danger: predicted impact in {soonest.predicted_seconds:.1f}s. Clear workers from path now."
                )
            else:
                messages.append(
                    f"Potential collision in {soonest.predicted_seconds:.1f}s. Slow moving equipment and clear route."
                )

        if near_miss_events:
            near = min(near_miss_events, key=lambda e: e.ttc_seconds)
            messages.append(
                f"Near-miss risk: tracks #{near.object_a} and #{near.object_b} are too close. Increase separation now."
            )

        proximity_hazards = [
            h for h in scene_hazard_events if h.hazard_type in {"worker_equipment_proximity", "dangerous_object_proximity"}
        ]
        if proximity_hazards:
            top_proximity = max(proximity_hazards, key=lambda h: h.severity * h.confidence)
            messages.append(f"Danger nearby: {top_proximity.details}. Keep clear and secure the area.")

        if ppe_violations:
            missing = sorted({item for v in ppe_violations for item in v.missing_items})
            messages.append(f"PPE alert: missing {', '.join(missing)}. Correct immediately.")

        if behavior_events:
            top = max(behavior_events, key=lambda e: e.severity)
            messages.append(f"Unsafe behavior detected: {top.event_type}. Slow down and follow protocol.")

        if fatigue_events:
            top = max(fatigue_events, key=lambda f: f.fatigue_score)
            messages.append(
                f"Fatigue warning for worker #{top.track_id}. Recommend short break and supervisor check."
            )

        other_scene_hazards = [
            h for h in scene_hazard_events if h.hazard_type not in {"worker_equipment_proximity", "dangerous_object_proximity"}
        ]
        if other_scene_hazards:
            top_scene = max(other_scene_hazards, key=lambda h: h.severity * h.confidence)
            messages.append(f"Scene hazard: {top_scene.hazard_type}. {top_scene.details}.")

        if risk_level == "HIGH":
            messages.append("High risk zone. Supervisor intervention recommended now.")

        if not messages:
            messages.append("Safety status normal. Continue monitoring.")

        return messages[:3]
