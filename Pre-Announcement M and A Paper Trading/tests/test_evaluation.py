import numpy as np

from src.evaluation import lift_at_fraction, rare_event_metrics


def test_lift_top_fraction():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    assert lift_at_fraction(y, score, 0.1) == 10


def test_rare_event_metrics_base_rate():
    metrics = rare_event_metrics(np.array([1, 0, 0, 0]), np.array([0.9, 0.2, 0.1, 0.0]))
    assert metrics["base_event_rate"] == 0.25
    assert metrics["pr_auc"] >= 0.25
