from src.config import RiskWeights, Thresholds


class RiskScorer:
    def __init__(self, weights: RiskWeights, thresholds: Thresholds):
        self.weights = weights
        self.thresholds = thresholds

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def compute(
        self,
        ppe_component: float,
        behavior_component: float,
        near_miss_component: float,
        collision_component: float,
        predicted_hazard_component: float,
        fatigue_component: float,
        surroundings_component: float,
    ):
        score = (
            self.weights.ppe_non_compliance * self._clamp(ppe_component)
            + self.weights.unsafe_behavior * self._clamp(behavior_component)
            + self.weights.near_miss * self._clamp(near_miss_component)
            + self.weights.collision_risk * self._clamp(collision_component)
            + self.weights.hazard_prediction * self._clamp(predicted_hazard_component)
            + self.weights.fatigue_risk * self._clamp(fatigue_component)
            + self.weights.surroundings_risk * self._clamp(surroundings_component)
        )
        score = self._clamp(score)

        if score >= self.thresholds.high_risk_score:
            level = "HIGH"
        elif score >= self.thresholds.medium_risk_score:
            level = "MEDIUM"
        else:
            level = "LOW"

        return score, level
