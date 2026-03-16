from dataclasses import dataclass, field


@dataclass
class RiskWeights:
    ppe_non_compliance: float = 0.22
    unsafe_behavior: float = 0.16
    near_miss: float = 0.14
    collision_risk: float = 0.14
    hazard_prediction: float = 0.14
    fatigue_risk: float = 0.10
    surroundings_risk: float = 0.10


@dataclass
class Thresholds:
    near_miss_distance_px: float = 70.0
    ttc_seconds_threshold: float = 2.5
    collision_horizon_seconds: float = 3.0
    fatigue_stationary_seconds: float = 30.0
    fatigue_movement_threshold_px_s: float = 6.0
    fatigue_alert_score: float = 0.65
    high_risk_score: float = 0.75
    medium_risk_score: float = 0.45


@dataclass
class PPEPolicy:
    required_for_person = ("helmet", "vest", "gloves", "goggles", "boots")


@dataclass
class AppConfig:
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    ppe_policy: PPEPolicy = field(default_factory=PPEPolicy)
