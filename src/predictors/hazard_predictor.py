from collections import deque
from typing import Deque


class HazardPredictor:
    """
    Temporal hazard estimator.
    Replace this with your trained ML model once data is available.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._risk_history: Deque[float] = deque(maxlen=window_size)

    def predict(self, instantaneous_risk: float) -> float:
        self._risk_history.append(instantaneous_risk)
        if not self._risk_history:
            return 0.0

        avg = sum(self._risk_history) / len(self._risk_history)
        trend = 0.0
        if len(self._risk_history) >= 2:
            trend = self._risk_history[-1] - self._risk_history[0]

        predicted = avg + 0.35 * max(0.0, trend)
        return max(0.0, min(1.0, predicted))
