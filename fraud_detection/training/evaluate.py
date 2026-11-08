"""Threshold selection, calibration, uncertainty, and fraud-business metrics.

The C++ engine intentionally evaluates only the exported neural network.  Its
sigmoid output is therefore a *raw risk score*, especially because the network
is trained with a class-weighted loss.  This module fits and applies the small
post-hoc calibration layer used by Python callers and records uncertainty for
held-out evaluation; it never changes the portable ``.mie`` network.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression


CALIBRATION_FORMAT = "platt_logit_v1"
IDENTITY_CALIBRATION: dict[str, Any] = {
    "format": CALIBRATION_FORMAT,
    "method": "identity",
    "input": "raw_sigmoid_score",
}


def _as_binary_target(target: np.ndarray) -> np.ndarray:
    values = np.asarray(target, dtype=np.int64).reshape(-1)
    if values.size == 0 or not np.isin(values, [0, 1]).all():
        raise ValueError("target must contain one or more binary 0/1 labels")
    if np.unique(values).size != 2:
        raise ValueError("target must contain both classes")
    return values


def _as_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("scores must be finite values between zero and one")
    return values


def fit_platt_calibration(target: np.ndarray, raw_scores: np.ndarray) -> dict[str, Any]:
    """Fit Platt calibration on validation data without touching the test set.

    A logistic regression is learned over the logits of raw sigmoid scores.
    Clipping only protects the logit transform from an exact zero or one and
    has no material effect on ordinary model outputs.
    """

    labels = _as_binary_target(target)
    scores = _as_scores(raw_scores)
    if labels.shape != scores.shape:
        raise ValueError("target and raw_scores must have the same length")
    epsilon = 1e-6
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    # This is the standard Platt scaling objective.  A fixed deterministic
    # solver makes saved model metadata reproducible for a given validation set.
    calibrator = LogisticRegression(C=1_000_000.0, solver="lbfgs", max_iter=1_000)
    calibrator.fit(logits, labels)
    return {
        "format": CALIBRATION_FORMAT,
        "method": "platt_logit",
        "input": "logit_of_raw_sigmoid_score",
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }


def calibrate_risk_scores(raw_scores: np.ndarray, calibration: dict[str, Any] | None = None) -> np.ndarray:
    """Convert raw engine scores to calibrated fraud probabilities.

    Missing calibration deliberately means identity transformation so old model
    artifacts remain usable.  Invalid metadata raises an explicit error rather
    than silently producing misleading risk bands in the UI.
    """

    scores = _as_scores(raw_scores)
    settings = calibration or IDENTITY_CALIBRATION
    method = settings.get("method", "identity")
    if method == "identity":
        return scores.astype(np.float32, copy=False)
    if settings.get("format") != CALIBRATION_FORMAT or method != "platt_logit":
        raise ValueError("Unsupported score-calibration metadata")
    try:
        coefficient = float(settings["coefficient"])
        intercept = float(settings["intercept"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Platt calibration metadata is incomplete") from exc
    if not np.isfinite(coefficient) or not np.isfinite(intercept):
        raise ValueError("Platt calibration metadata must be finite")
    epsilon = 1e-6
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped))
    transformed = coefficient * logits + intercept
    # Branch form avoids overflow when a future model produces large logits.
    probabilities = np.where(
        transformed >= 0.0,
        1.0 / (1.0 + np.exp(-transformed)),
        np.exp(transformed) / (1.0 + np.exp(transformed)),
    )
    return probabilities.astype(np.float32, copy=False)


def select_f1_threshold(target: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the validation threshold with maximum F1; ties retain the first."""

    labels = _as_binary_target(target)
    scores = _as_scores(probabilities)
    if labels.shape != scores.shape:
        raise ValueError("target and probabilities must have the same length")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    f1_values = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1_values))])


def classification_metrics(target: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Calculate all requested metrics using an explicitly supplied threshold."""

    labels = _as_binary_target(target)
    scores = _as_scores(probabilities)
    if labels.shape != scores.shape:
        raise ValueError("target and probabilities must have the same length")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite value between zero and one")
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).astype(int).tolist(),
    }


def select_cost_threshold(
    target: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    *,
    false_positive_review_cost: float = 10.0,
) -> float:
    """Choose a validation threshold that minimizes a stated review-cost policy.

    A missed fraud loses its transaction ``Amount``; flagging a genuine
    transaction incurs the fixed manual-review cost.  True positives are
    assumed to be stopped before the amount is lost.  The policy is deliberately
    explicit so a deployment can replace the $10 review assumption.
    """

    labels = _as_binary_target(target)
    scores = _as_scores(probabilities)
    transaction_amounts = np.asarray(amounts, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.shape != transaction_amounts.shape:
        raise ValueError("target, probabilities, and amounts must have the same length")
    if not np.isfinite(transaction_amounts).all() or np.any(transaction_amounts < 0.0):
        raise ValueError("amounts must be finite, non-negative values")
    if not np.isfinite(false_positive_review_cost) or false_positive_review_cost < 0.0:
        raise ValueError("false_positive_review_cost must be finite and non-negative")

    # Include zero and one so a policy can choose "flag none" or "flag all".
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold = float(candidates[0])
    best_cost = float("inf")
    for threshold in candidates:
        predictions = scores >= threshold
        false_negatives = (labels == 1) & ~predictions
        false_positives = (labels == 0) & predictions
        cost = float(transaction_amounts[false_negatives].sum() +
                     false_positive_review_cost * false_positives.sum())
        # A higher threshold is preferable when policies tie: fewer alerts.
        if cost < best_cost - 1e-12 or (abs(cost - best_cost) <= 1e-12 and threshold > best_threshold):
            best_threshold, best_cost = float(threshold), cost
    return best_threshold


def cost_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    *,
    false_positive_review_cost: float = 10.0,
) -> dict[str, float]:
    """Report the transparent financial assumptions behind a decision policy."""

    labels = _as_binary_target(target)
    scores = _as_scores(probabilities)
    transaction_amounts = np.asarray(amounts, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.shape != transaction_amounts.shape:
        raise ValueError("target, probabilities, and amounts must have the same length")
    if not np.isfinite(transaction_amounts).all() or np.any(transaction_amounts < 0.0):
        raise ValueError("amounts must be finite, non-negative values")
    if not np.isfinite(false_positive_review_cost) or false_positive_review_cost < 0.0:
        raise ValueError("false_positive_review_cost must be finite and non-negative")
    predictions = scores >= threshold
    caught = (labels == 1) & predictions
    missed = (labels == 1) & ~predictions
    false_positives = (labels == 0) & predictions
    baseline_loss = float(transaction_amounts[labels == 1].sum())
    missed_fraud_loss = float(transaction_amounts[missed].sum())
    review_cost = float(false_positive_review_cost * false_positives.sum())
    intervention_cost = missed_fraud_loss + review_cost
    return {
        "false_positive_review_cost": float(false_positive_review_cost),
        "caught_fraud_count": float(caught.sum()),
        "caught_fraud_amount": float(transaction_amounts[caught].sum()),
        "missed_fraud_count": float(missed.sum()),
        "missed_fraud_amount": missed_fraud_loss,
        "false_positive_count": float(false_positives.sum()),
        "false_positive_review_total": review_cost,
        "no_intervention_fraud_loss": baseline_loss,
        "estimated_intervention_cost": intervention_cost,
        "estimated_net_money_saved": baseline_loss - intervention_cost,
    }


def bootstrap_confidence_intervals(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Return stratified percentile bootstrap intervals for held-out metrics.

    Resampling positives and negatives independently preserves the rare-event
    rate in every replicate, avoiding invalid one-class PR/ROC samples.
    """

    labels = _as_binary_target(target)
    scores = _as_scores(probabilities)
    if labels.shape != scores.shape:
        raise ValueError("target and probabilities must have the same length")
    if resamples < 1:
        raise ValueError("resamples must be at least one")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    point = classification_metrics(labels, scores, threshold)
    names = ("precision", "recall", "f1", "pr_auc", "roc_auc")
    samples: dict[str, list[float]] = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    for _ in range(resamples):
        indices = np.concatenate((
            rng.choice(positive_indices, size=positive_indices.size, replace=True),
            rng.choice(negative_indices, size=negative_indices.size, replace=True),
        ))
        # Shuffle makes the sample representative if a future metric depends on
        # ordering, while not affecting the current classification metrics.
        rng.shuffle(indices)
        replicate = classification_metrics(labels[indices], scores[indices], threshold)
        for name in names:
            samples[name].append(float(replicate[name]))

    alpha = (1.0 - confidence) / 2.0
    return {
        name: {
            "point_estimate": float(point[name]),
            "lower": float(np.quantile(samples[name], alpha)),
            "upper": float(np.quantile(samples[name], 1.0 - alpha)),
            "confidence": float(confidence),
            "resamples": float(resamples),
        }
        for name in names
    }
