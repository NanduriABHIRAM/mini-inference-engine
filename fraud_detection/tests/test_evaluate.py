from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_detection.training.evaluate import (
    bootstrap_confidence_intervals,
    calibrate_risk_scores,
    cost_metrics,
    fit_platt_calibration,
    select_cost_threshold,
)
from fraud_detection.training.preprocess import FEATURE_COLUMNS, TARGET_COLUMN
from fraud_detection.training.train import split_train_validation_test


def test_calibration_and_bootstrap_keep_probabilities_well_formed() -> None:
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.01, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.99])
    calibration = fit_platt_calibration(target, scores)
    calibrated = calibrate_risk_scores(scores, calibration)
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))
    intervals = bootstrap_confidence_intervals(target, calibrated, 0.5, resamples=10, seed=1)
    assert set(intervals) == {"precision", "recall", "f1", "pr_auc", "roc_auc"}
    assert intervals["pr_auc"]["lower"] <= intervals["pr_auc"]["upper"]


def test_cost_threshold_reports_money_saved_with_explicit_review_cost() -> None:
    target = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.4, 0.6, 0.1])
    amounts = np.array([100.0, 40.0, 5.0, 5.0])
    threshold = select_cost_threshold(target, probabilities, amounts, false_positive_review_cost=10.0)
    report = cost_metrics(target, probabilities, amounts, threshold, false_positive_review_cost=10.0)
    assert 0.0 <= threshold <= 1.0
    assert report["estimated_net_money_saved"] >= 0.0


def test_chronological_split_never_trains_on_future_rows() -> None:
    rows = []
    for index in range(30):
        row = {feature: float(index) for feature in FEATURE_COLUMNS}
        row["Time"] = float(index)
        row[TARGET_COLUMN] = index % 2
        rows.append(row)
    train, validation, test = split_train_validation_test(pd.DataFrame(rows), strategy="chronological", seed=42)
    assert train["Time"].max() <= validation["Time"].min()
    assert validation["Time"].max() <= test["Time"].min()
