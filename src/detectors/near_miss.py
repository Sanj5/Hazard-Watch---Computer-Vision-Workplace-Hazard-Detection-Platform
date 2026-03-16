from collections import defaultdict
from typing import Dict, List, Tuple
import math

from src.config import Thresholds
from src.utils.types import Detection, NearMissEvent, CollisionRiskEvent


class NearMissAndCollisionDetector:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self._last_centers: Dict[int, Tuple[float, float]] = {}
        self._last_time: float = 0.0
        self._velocity_cache = defaultdict(lambda: (0.0, 0.0))

    @staticmethod
    def _center(bbox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _update_velocities(self, detections: List[Detection], dt: float) -> Dict[int, Tuple[float, float]]:
        centers = {d.track_id: self._center(d.bbox) for d in detections}
        if dt <= 1e-6:
            return dict(self._velocity_cache)

        for track_id, c in centers.items():
            if track_id in self._last_centers:
                px, py = self._last_centers[track_id]
                self._velocity_cache[track_id] = ((c[0] - px) / dt, (c[1] - py) / dt)
        self._last_centers = centers
        return dict(self._velocity_cache)

    def analyze(
        self,
        detections: List[Detection],
        dt_seconds: float,
    ) -> Tuple[List[NearMissEvent], List[CollisionRiskEvent]]:
        moving = [d for d in detections if d.label in {"person", "forklift", "truck", "robot"}]
        velocities = self._update_velocities(moving, dt_seconds)

        near_misses: List[NearMissEvent] = []
        collisions: List[CollisionRiskEvent] = []

        for i in range(len(moving)):
            for j in range(i + 1, len(moving)):
                a = moving[i]
                b = moving[j]
                ax, ay = self._center(a.bbox)
                bx, by = self._center(b.bbox)
                dist = math.dist((ax, ay), (bx, by))

                avx, avy = velocities.get(a.track_id, (0.0, 0.0))
                bvx, bvy = velocities.get(b.track_id, (0.0, 0.0))
                rvx, rvy = avx - bvx, avy - bvy
                rx, ry = ax - bx, ay - by
                rel_speed_sq = rvx * rvx + rvy * rvy

                ttc = 999.0
                if rel_speed_sq > 1e-6:
                    ttc = -((rx * rvx + ry * rvy) / rel_speed_sq)

                if dist <= self.thresholds.near_miss_distance_px and 0.0 <= ttc <= self.thresholds.ttc_seconds_threshold:
                    confidence = max(0.5, min(1.0, (self.thresholds.near_miss_distance_px / (dist + 1.0))))
                    near_misses.append(
                        NearMissEvent(
                            object_a=a.track_id,
                            object_b=b.track_id,
                            distance_px=dist,
                            ttc_seconds=ttc,
                            confidence=confidence,
                        )
                    )

                if 0.0 <= ttc <= self.thresholds.collision_horizon_seconds:
                    collisions.append(
                        CollisionRiskEvent(
                            object_a=a.track_id,
                            object_b=b.track_id,
                            predicted_seconds=ttc,
                            confidence=max(0.4, min(1.0, 1.0 - ttc / self.thresholds.collision_horizon_seconds)),
                        )
                    )

        return near_misses, collisions
