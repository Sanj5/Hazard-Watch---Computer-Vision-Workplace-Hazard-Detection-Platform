from src.config import RiskWeights, Thresholds
from src.risk.risk_scorer import RiskScorer


def test_high_risk_classification():
    scorer = RiskScorer(RiskWeights(), Thresholds())
    score, level = scorer.compute(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert score == 1.0
    assert level == "HIGH"


def test_low_risk_classification():
    scorer = RiskScorer(RiskWeights(), Thresholds())
    score, level = scorer.compute(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert score == 0.0
    assert level == "LOW"


def test_surroundings_contribute_to_risk():
    scorer = RiskScorer(RiskWeights(), Thresholds())
    score, level = scorer.compute(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert score > 0.0
    assert level in {"LOW", "MEDIUM", "HIGH"}
