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

        if ppe_violations:
            missing = sorted({item for v in ppe_violations for item in v.missing_items})
            messages.append(f"PPE alert: missing {', '.join(missing)}. Correct immediately.")

        if behavior_events:
            top = max(behavior_events, key=lambda e: e.severity)
            messages.append(f"Unsafe behavior detected: {top.event_type}. Slow down and follow protocol.")

        if near_miss_events:
            messages.append("Near-miss detected. Increase separation and reduce speed in the area.")

        if collision_events:
            soonest = min(collision_events, key=lambda c: c.predicted_seconds)
            messages.append(f"Potential collision in {soonest.predicted_seconds:.1f}s. Stop and reassess route.")

        if fatigue_events:
            top = max(fatigue_events, key=lambda f: f.fatigue_score)
            messages.append(
                f"Fatigue warning for worker #{top.track_id}. Recommend short break and supervisor check."
            )

        if scene_hazard_events:
            top_scene = max(scene_hazard_events, key=lambda h: h.severity * h.confidence)
            messages.append(f"Scene hazard: {top_scene.hazard_type}. {top_scene.details}.")

        if risk_level == "HIGH":
            messages.append("High risk zone. Supervisor intervention recommended now.")

        if not messages:
            messages.append("Safety status normal. Continue monitoring.")

        return messages[:3]
