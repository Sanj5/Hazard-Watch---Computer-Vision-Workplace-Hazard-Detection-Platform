from typing import List

from src.config import AppConfig
from src.detectors.ppe_detector import PPEDetector
from src.detectors.behavior_monitor import BehaviorMonitor
from src.detectors.fatigue_monitor import FatigueMonitor
from src.detectors.surroundings_monitor import SurroundingsMonitor
from src.detectors.near_miss import NearMissAndCollisionDetector
from src.predictors.hazard_predictor import HazardPredictor
from src.predictors.collision_predictor import FutureCollisionPredictor
from src.risk.risk_scorer import RiskScorer
from src.coaching.safety_coach import SafetyCoach
from src.pose.mediapipe_pose import MediaPipePoseBehaviorAnalyzer
from src.utils.types import Detection, FrameAnalysis


class SafetyPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.ppe_detector = PPEDetector(config.ppe_policy.required_for_person)
        self.behavior_monitor = BehaviorMonitor()
        self.fatigue_monitor = FatigueMonitor(
            movement_threshold_px_s=config.thresholds.fatigue_movement_threshold_px_s,
            stationary_alert_seconds=config.thresholds.fatigue_stationary_seconds,
            min_score_for_alert=config.thresholds.fatigue_alert_score,
        )
        self.near_miss_detector = NearMissAndCollisionDetector(config.thresholds)
        self.surroundings_monitor = SurroundingsMonitor()
        self.future_collision_predictor = FutureCollisionPredictor(config.thresholds)
        self.pose_behavior = MediaPipePoseBehaviorAnalyzer(enabled=True)
        self.hazard_predictor = HazardPredictor()
        self.risk_scorer = RiskScorer(config.risk_weights, config.thresholds)
        self.coach = SafetyCoach()

    def analyze_frame(self, detections: List[Detection], dt_seconds: float, frame=None) -> FrameAnalysis:
        ppe_violations = self.ppe_detector.detect_violations(detections)
        pose_events = self.pose_behavior.analyze(frame)
        behavior_events = self.behavior_monitor.analyze(detections, extra_events=pose_events)
        fatigue_events = self.fatigue_monitor.analyze(detections, behavior_events, dt_seconds)
        scene_hazard_events = self.surroundings_monitor.analyze(detections)
        near_miss_events, collision_events_velocity = self.near_miss_detector.analyze(detections, dt_seconds)
        collision_events_forecast = self.future_collision_predictor.predict(detections, dt_seconds)
        collision_events = collision_events_velocity + collision_events_forecast

        ppe_component = min(1.0, len(ppe_violations) / 4.0)
        behavior_component = max([0.0] + [e.severity * e.confidence for e in behavior_events])
        near_miss_component = max([0.0] + [e.confidence for e in near_miss_events])
        collision_component = max([0.0] + [e.confidence for e in collision_events])
        fatigue_component = max([0.0] + [e.fatigue_score * e.confidence for e in fatigue_events])
        surroundings_component = max([0.0] + [e.severity * e.confidence for e in scene_hazard_events])

        instantaneous = max(
            ppe_component,
            behavior_component,
            near_miss_component,
            collision_component,
            surroundings_component,
        )
        predicted_hazard = self.hazard_predictor.predict(instantaneous)

        risk_score, risk_level = self.risk_scorer.compute(
            ppe_component,
            behavior_component,
            near_miss_component,
            collision_component,
            predicted_hazard,
            fatigue_component,
            surroundings_component,
        )

        critical_component = max(
            behavior_component,
            near_miss_component,
            collision_component,
            surroundings_component,
        )
        if critical_component >= 0.9:
            risk_score = max(risk_score, 0.8)
            risk_level = "HIGH"
        elif critical_component >= 0.7 and risk_level == "LOW":
            risk_score = max(risk_score, 0.5)
            risk_level = "MEDIUM"

        coaching_messages = self.coach.generate(
            ppe_violations,
            behavior_events,
            near_miss_events,
            collision_events,
            fatigue_events,
            scene_hazard_events,
            risk_level,
        )

        return FrameAnalysis(
            ppe_violations=ppe_violations,
            behavior_events=behavior_events,
            near_miss_events=near_miss_events,
            collision_events=collision_events,
            fatigue_events=fatigue_events,
            scene_hazard_events=scene_hazard_events,
            predicted_hazard_probability=predicted_hazard,
            risk_score=risk_score,
            risk_level=risk_level,
            coaching_messages=coaching_messages,
            metrics={
                "ppe_component": ppe_component,
                "behavior_component": behavior_component,
                "near_miss_component": near_miss_component,
                "collision_component": collision_component,
                "fatigue_component": fatigue_component,
                "surroundings_component": surroundings_component,
            },
        )
