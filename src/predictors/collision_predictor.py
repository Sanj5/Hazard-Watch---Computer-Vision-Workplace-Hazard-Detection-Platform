from typing import List
import math

from src.config import Thresholds
from src.predictors.trajectory_lstm import TrajectoryForecaster
from src.tracking.trajectory_buffer import TrajectoryBuffer
from src.utils.types import CollisionRiskEvent, Detection


class FutureCollisionPredictor:
    def __init__(self, thresholds: Thresholds, forecast_steps: int = 8):
        self.thresholds = thresholds
        self.traj = TrajectoryBuffer(maxlen=40)
        self.forecaster = TrajectoryForecaster(pred_steps=forecast_steps)
        self.forecast_steps = forecast_steps

    def predict(self, detections: List[Detection], dt_seconds: float) -> List[CollisionRiskEvent]:
        moving = [d for d in detections if d.label in {"person", "forklift", "truck", "robot"}]
        self.traj.update(moving)

        forecasts = {}
        for d in moving:
            history = self.traj.get_history(d.track_id)
            forecasts[d.track_id] = self.forecaster.predict(history)

        events: List[CollisionRiskEvent] = []
        horizon_dist = self.thresholds.near_miss_distance_px * 1.1

        for i in range(len(moving)):
            for j in range(i + 1, len(moving)):
                a_id = moving[i].track_id
                b_id = moving[j].track_id
                a_forecast = forecasts.get(a_id, [])
                b_forecast = forecasts.get(b_id, [])
                if not a_forecast or not b_forecast:
                    continue

                min_dist = float("inf")
                min_step = 0
                for k, (pa, pb) in enumerate(zip(a_forecast, b_forecast)):
                    dist = math.dist(pa, pb)
                    if dist < min_dist:
                        min_dist = dist
                        min_step = k + 1

                if min_dist <= horizon_dist:
                    pred_sec = max(0.001, min_step * max(0.001, dt_seconds))
                    confidence = max(0.4, min(1.0, 1.0 - min_dist / (horizon_dist + 1.0)))
                    events.append(
                        CollisionRiskEvent(
                            object_a=a_id,
                            object_b=b_id,
                            predicted_seconds=pred_sec,
                            confidence=confidence,
                        )
                    )

        return events
