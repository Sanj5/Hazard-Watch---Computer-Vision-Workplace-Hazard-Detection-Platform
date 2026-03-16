from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Detection:
    track_id: int
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]


@dataclass
class PPEViolation:
    person_track_id: int
    missing_items: List[str]
    confidence: float


@dataclass
class BehaviorEvent:
    event_type: str
    track_id: int
    confidence: float
    severity: float


@dataclass
class NearMissEvent:
    object_a: int
    object_b: int
    distance_px: float
    ttc_seconds: float
    confidence: float


@dataclass
class CollisionRiskEvent:
    object_a: int
    object_b: int
    predicted_seconds: float
    confidence: float


@dataclass
class FatigueEvent:
    track_id: int
    fatigue_score: float
    stationary_seconds: float
    confidence: float


@dataclass
class SceneHazardEvent:
    hazard_type: str
    confidence: float
    severity: float
    details: str


@dataclass
class FrameAnalysis:
    ppe_violations: List[PPEViolation] = field(default_factory=list)
    behavior_events: List[BehaviorEvent] = field(default_factory=list)
    near_miss_events: List[NearMissEvent] = field(default_factory=list)
    collision_events: List[CollisionRiskEvent] = field(default_factory=list)
    fatigue_events: List[FatigueEvent] = field(default_factory=list)
    scene_hazard_events: List[SceneHazardEvent] = field(default_factory=list)
    predicted_hazard_probability: float = 0.0
    risk_score: float = 0.0
    risk_level: str = "LOW"
    coaching_messages: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
