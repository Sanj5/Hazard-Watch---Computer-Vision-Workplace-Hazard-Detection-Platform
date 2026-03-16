from typing import List

from src.utils.types import BehaviorEvent

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


class MediaPipePoseBehaviorAnalyzer:
    """Detect basic unsafe posture from torso tilt; extend this for richer behavior models."""

    def __init__(self, enabled: bool = True):
        self.available = False
        self.enabled = enabled and mp is not None
        self._pose = None

        if not self.enabled:
            return

        # Some recent mediapipe wheels expose only `tasks` and not `solutions`.
        try:
            solutions = getattr(mp, "solutions", None)
            if solutions is not None and hasattr(solutions, "pose"):
                self._pose = solutions.pose.Pose()
                self.available = True
            else:
                self.enabled = False
        except Exception:
            self.enabled = False

    def analyze(self, frame) -> List[BehaviorEvent]:
        if not self.enabled or self._pose is None or frame is None:
            return []

        rgb = frame[:, :, ::-1]
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return []

        landmarks = result.pose_landmarks.landmark
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
        hip_x = (left_hip.x + right_hip.x) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0

        dx = shoulder_x - hip_x
        dy = abs(shoulder_y - hip_y) + 1e-6
        tilt_ratio = abs(dx) / dy

        if tilt_ratio > 0.35:
            confidence = min(1.0, 0.55 + tilt_ratio)
            return [
                BehaviorEvent(
                    event_type="unsafe_posture",
                    track_id=-1,
                    confidence=confidence,
                    severity=0.65,
                )
            ]
        return []
