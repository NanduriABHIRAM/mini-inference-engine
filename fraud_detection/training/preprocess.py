"""Shared, leakage-safe preprocessing for training and the fraud dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["Time", *[f"V{index}" for index in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
PREPROCESSOR_FORMAT = "fraud_preprocessor_v1"
TIME_FEATURES = {"raw", "hour_of_day"}


class InputSchemaError(ValueError):
    """Raised when a fraud transaction CSV cannot be safely used for inference."""


def validate_transaction_frame(frame: pd.DataFrame, *, require_target: bool = False) -> pd.DataFrame:
    """Return finite features in the one persisted training order.

    Uploaded inference files may include ``Class`` for later comparison, but no
    other columns are accepted. Rejecting extras prevents a silently shifted
    feature vector from reaching the C++ engine.
    """

    if frame.empty:
        raise InputSchemaError("The CSV contains no transaction rows.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise InputSchemaError(f"The CSV has duplicate columns: {duplicates}")

    expected = set(FEATURE_COLUMNS)
    available = set(frame.columns)
    missing = [column for column in FEATURE_COLUMNS if column not in available]
    unexpected = [column for column in frame.columns if column not in expected | {TARGET_COLUMN}]
    if missing:
        raise InputSchemaError(f"The CSV is missing required features: {', '.join(missing)}")
    if unexpected:
        raise InputSchemaError(f"The CSV has unsupported columns: {', '.join(unexpected)}")
    if require_target and TARGET_COLUMN not in available:
        raise InputSchemaError("Training data must include the Class target column.")

    try:
        features = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise InputSchemaError("All transaction feature values must be numeric.") from exc
    values = features.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise InputSchemaError("Transaction features must be finite values.")

    result = features.copy()
    if TARGET_COLUMN in available:
        try:
            target = pd.to_numeric(frame[TARGET_COLUMN], errors="raise")
        except (TypeError, ValueError) as exc:
            raise InputSchemaError("Class must contain only 0 (genuine) or 1 (fraud).") from exc
        target_values = target.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(target_values).all() or not target.isin([0, 1]).all():
            raise InputSchemaError("Class must contain only 0 (genuine) or 1 (fraud).")
        result[TARGET_COLUMN] = target.astype(np.int64)
    return result


def load_transactions(path: str | Path, *, require_target: bool = False) -> pd.DataFrame:
    """Read a raw Kaggle-format CSV and validate it before preprocessing."""

    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise InputSchemaError(f"Could not read transaction CSV: {exc}") from exc
    return validate_transaction_frame(frame, require_target=require_target)


def _prepare_features(frame: pd.DataFrame, time_feature: str) -> pd.DataFrame:
    """Validate, order, and optionally make the timestamp cyclically useful.

    The Kaggle ``Time`` column is elapsed seconds from the first transaction.
    Its raw value can leak chronological position when train/test are split in
    time order.  ``hour_of_day`` keeps a useful daily-cycle signal while
    avoiding the absolute timestamp trend, and retains the 30-feature engine
    contract.
    """

    if time_feature not in TIME_FEATURES:
        raise InputSchemaError("time_feature must be 'raw' or 'hour_of_day'.")
    checked = validate_transaction_frame(frame, require_target=False)
    features = checked.loc[:, FEATURE_COLUMNS].copy()
    if time_feature == "hour_of_day":
        features["Time"] = np.mod(features["Time"].to_numpy(dtype=np.float64), 86_400.0) / 3_600.0
    return features


def fit_preprocessor(
    training_features: pd.DataFrame, *, time_feature: Literal["raw", "hour_of_day"] = "raw"
) -> dict[str, Any]:
    """Fit scaling only on training rows and return JSON-safe metadata.

    ``raw`` is retained as the default for compatibility with already-trained
    artifacts. New chronological training runs explicitly use ``hour_of_day``.
    """

    checked = _prepare_features(training_features, time_feature)
    scaler = StandardScaler()
    scaler.fit(checked)
    return {
        "format": PREPROCESSOR_FORMAT,
        "feature_columns": FEATURE_COLUMNS,
        "time_feature": time_feature,
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
    }


def transform_features(frame: pd.DataFrame, preprocessor: dict[str, Any]) -> np.ndarray:
    """Apply the persisted training scaler and return FP32 engine inputs."""

    if preprocessor.get("format") != PREPROCESSOR_FORMAT:
        raise InputSchemaError("Unsupported preprocessing artifact format.")
    if preprocessor.get("feature_columns") != FEATURE_COLUMNS:
        raise InputSchemaError("Preprocessing artifact does not match the fraud feature order.")
    mean = np.asarray(preprocessor.get("mean"), dtype=np.float64)
    scale = np.asarray(preprocessor.get("scale"), dtype=np.float64)
    if mean.shape != (len(FEATURE_COLUMNS),) or scale.shape != (len(FEATURE_COLUMNS),):
        raise InputSchemaError("Preprocessing artifact has invalid scaler dimensions.")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise InputSchemaError("Preprocessing artifact has invalid scaler values.")

    time_feature = preprocessor.get("time_feature", "raw")
    checked = _prepare_features(frame, time_feature)
    values = checked.to_numpy(dtype=np.float64, copy=False)
    transformed = (values - mean) / scale
    if not np.isfinite(transformed).all():
        raise InputSchemaError("Preprocessing produced non-finite feature values.")
    return transformed.astype(np.float32, copy=False)


def save_preprocessor(preprocessor: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(preprocessor, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_preprocessor(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputSchemaError(f"Could not load preprocessing artifact: {exc}") from exc
    # Reuse transform validation so corrupt metadata fails before a UI run.
    transform_features(pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS), payload)
    return payload
