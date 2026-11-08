"""Train, evaluate, and export the fraud model for the C++ inference engine."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_detection.training.evaluate import (
    bootstrap_confidence_intervals,
    calibrate_risk_scores,
    classification_metrics,
    cost_metrics,
    fit_platt_calibration,
    select_cost_threshold,
    select_f1_threshold,
)
from fraud_detection.training.model import MODEL_DIMENSIONS, FraudMLP, export_engine_model
from fraud_detection.training.preprocess import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    fit_preprocessor,
    load_transactions,
    save_preprocessor,
    transform_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "fraud_detection" / "data" / "creditcard.csv"
DEFAULT_MODELS = PROJECT_ROOT / "fraud_detection" / "models"
DEFAULT_RESULTS = PROJECT_ROOT / "fraud_detection" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--split-strategy",
        choices=["chronological", "stratified_random"],
        default="chronological",
        help="Use chronological deployment-like evaluation by default; retain stratified_random for comparison.",
    )
    parser.add_argument(
        "--time-feature",
        choices=["hour_of_day", "raw"],
        default="hour_of_day",
        help="Representation of the dataset Time feature; hour_of_day avoids using absolute chronology as a feature.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument(
        "--false-positive-review-cost",
        type=float,
        default=10.0,
        help="Assumed manual-review cost used for the validation cost policy.",
    )
    parser.add_argument("--skip-baselines", action="store_true", help="Skip the Logistic Regression and HistGradientBoosting comparisons.")
    parser.add_argument("--baseline-hgb-iterations", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def probabilities(model: FraudMLP, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features).to(device)
        return torch.sigmoid(model(tensor)).squeeze(1).cpu().numpy()


def split_train_validation_test(
    frame: pd.DataFrame,
    *,
    strategy: str,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Produce leakage-safe train/validation/test partitions.

    The default chronological split mirrors an operational setting: train on
    transactions available in the past, tune on the next period, then report
    only on a future holdout. ``stratified_random`` remains available for an
    apples-to-apples comparison with earlier experiments, not as the default.
    """

    if strategy not in {"chronological", "stratified_random"}:
        raise ValueError("strategy must be chronological or stratified_random")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must be positive and leave a test partition")
    if TARGET_COLUMN not in frame or "Time" not in frame:
        raise ValueError("frame must include Time and Class columns")
    if len(frame) < 6:
        raise ValueError("at least six rows are needed to make three partitions")

    if strategy == "chronological":
        ordered = frame.sort_values("Time", kind="mergesort").reset_index(drop=False)
        train_end = int(len(ordered) * train_fraction)
        validation_end = train_end + int(len(ordered) * validation_fraction)
        train, validation, test = (
            ordered.iloc[:train_end].copy(),
            ordered.iloc[train_end:validation_end].copy(),
            ordered.iloc[validation_end:].copy(),
        )
    else:
        train, holdout = train_test_split(
            frame, test_size=1.0 - train_fraction, stratify=frame[TARGET_COLUMN], random_state=seed
        )
        validation_share_of_holdout = validation_fraction / (1.0 - train_fraction)
        validation, test = train_test_split(
            holdout,
            test_size=1.0 - validation_share_of_holdout,
            stratify=holdout[TARGET_COLUMN],
            random_state=seed,
        )
        train, validation, test = train.copy(), validation.copy(), test.copy()

    for name, partition in (("train", train), ("validation", validation), ("test", test)):
        if partition.empty or partition[TARGET_COLUMN].nunique() != 2:
            raise ValueError(f"{name} split must contain both genuine and fraud transactions")
    return train, validation, test


def project_relative_path(path: Path) -> str:
    """Avoid writing an account-specific absolute path into evaluation.json."""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def partition_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "fraud_rows": int(frame[TARGET_COLUMN].sum()),
        "genuine_rows": int(len(frame) - frame[TARGET_COLUMN].sum()),
        "time_min": float(frame["Time"].min()),
        "time_max": float(frame["Time"].max()),
    }


def evaluate_scored_model(
    *,
    validation_target: np.ndarray,
    validation_raw_scores: np.ndarray,
    validation_amounts: np.ndarray,
    test_target: np.ndarray,
    test_raw_scores: np.ndarray,
    test_amounts: np.ndarray,
    false_positive_review_cost: float,
    bootstrap_resamples: int = 0,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], float, float, dict[str, Any]]:
    """Calibrate on validation, select policies there, and evaluate once on test."""

    calibration = fit_platt_calibration(validation_target, validation_raw_scores)
    validation_probabilities = calibrate_risk_scores(validation_raw_scores, calibration)
    test_probabilities = calibrate_risk_scores(test_raw_scores, calibration)
    f1_threshold = select_f1_threshold(validation_target, validation_probabilities)
    cost_threshold = select_cost_threshold(
        validation_target,
        validation_probabilities,
        validation_amounts,
        false_positive_review_cost=false_positive_review_cost,
    )
    validation_metrics = classification_metrics(validation_target, validation_probabilities, f1_threshold)
    test_metrics = classification_metrics(test_target, test_probabilities, f1_threshold)
    policy = {
        "threshold": cost_threshold,
        "false_positive_review_cost": false_positive_review_cost,
        "validation": cost_metrics(
            validation_target,
            validation_probabilities,
            validation_amounts,
            cost_threshold,
            false_positive_review_cost=false_positive_review_cost,
        ),
        "test": cost_metrics(
            test_target,
            test_probabilities,
            test_amounts,
            cost_threshold,
            false_positive_review_cost=false_positive_review_cost,
        ),
    }
    uncertainty: dict[str, Any] = {}
    if bootstrap_resamples:
        uncertainty = bootstrap_confidence_intervals(
            test_target,
            test_probabilities,
            f1_threshold,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        )
    return calibration, {"validation": validation_metrics, "test": test_metrics}, f1_threshold, cost_threshold, {
        "policy": policy,
        "uncertainty": uncertainty,
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
    }


def run_baselines(
    train_features: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    validation_amounts: np.ndarray,
    test_features: np.ndarray,
    test_target: np.ndarray,
    test_amounts: np.ndarray,
    *,
    seed: int,
    false_positive_review_cost: float,
    hgb_iterations: int,
) -> dict[str, Any]:
    """Fit reproducible, broadly available comparison baselines.

    Logistic Regression establishes a linear reference. HistGradientBoosting is
    a tree-based reference that captures non-linear interactions without adding
    a heavyweight optional dependency such as XGBoost.
    """

    if hgb_iterations < 1:
        raise ValueError("baseline_hgb_iterations must be positive")
    positive_weight = float((train_target == 0).sum() / (train_target == 1).sum())
    baseline_specs: list[tuple[str, Any, bool]] = [
        (
            "logistic_regression",
            LogisticRegression(class_weight="balanced", max_iter=1_000, solver="lbfgs", random_state=seed),
            False,
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=hgb_iterations,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=seed,
            ),
            True,
        ),
    ]
    reports: dict[str, Any] = {}
    sample_weights = np.where(train_target == 1, positive_weight, 1.0)
    for name, estimator, needs_sample_weights in baseline_specs:
        if needs_sample_weights:
            estimator.fit(train_features, train_target, sample_weight=sample_weights)
        else:
            estimator.fit(train_features, train_target)
        validation_raw_scores = estimator.predict_proba(validation_features)[:, 1]
        test_raw_scores = estimator.predict_proba(test_features)[:, 1]
        calibration, metrics, f1_threshold, cost_threshold, extras = evaluate_scored_model(
            validation_target=validation_target,
            validation_raw_scores=validation_raw_scores,
            validation_amounts=validation_amounts,
            test_target=test_target,
            test_raw_scores=test_raw_scores,
            test_amounts=test_amounts,
            false_positive_review_cost=false_positive_review_cost,
        )
        reports[name] = {
            "calibration": calibration,
            "classification_threshold": f1_threshold,
            "cost_policy": {"threshold": cost_threshold, **extras["policy"]},
            **metrics,
        }
    return reports


def main() -> None:
    args = parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.learning_rate <= 0
            or args.bootstrap_resamples < 0 or not 0.0 < args.bootstrap_confidence < 1.0
            or args.false_positive_review_cost < 0):
        raise SystemExit("epochs, batch-size, patience, and learning-rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but PyTorch cannot access CUDA")

    set_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    frame = load_transactions(args.data, require_target=True)
    if frame[TARGET_COLUMN].nunique() != 2:
        raise SystemExit("Training data must contain both genuine and fraudulent transactions")
    train_frame, validation_frame, test_frame = split_train_validation_test(
        frame, strategy=args.split_strategy, seed=args.seed
    )
    preprocessor = fit_preprocessor(train_frame, time_feature=args.time_feature)
    train_features = transform_features(train_frame, preprocessor)
    validation_features = transform_features(validation_frame, preprocessor)
    test_features = transform_features(test_frame, preprocessor)
    train_target = train_frame[TARGET_COLUMN].to_numpy(dtype=np.int64)
    validation_target = validation_frame[TARGET_COLUMN].to_numpy(dtype=np.int64)
    test_target = test_frame[TARGET_COLUMN].to_numpy(dtype=np.int64)

    train_dataset = TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_target.astype(np.float32)))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    model = FraudMLP().to(device)
    fraud_count = int(train_target.sum())
    genuine_count = int(len(train_target) - fraud_count)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([genuine_count / fraud_count], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_state: dict[str, torch.Tensor] | None = None
    best_pr_auc = -float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        cumulative_loss = 0.0
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features.to(device)).squeeze(1)
            loss = loss_function(logits, batch_target.to(device))
            loss.backward()
            optimizer.step()
            cumulative_loss += float(loss.detach().cpu()) * len(batch_target)

        validation_scores = probabilities(model, validation_features, device)
        validation_pr_auc = float(classification_metrics(validation_target, validation_scores, 0.5)["pr_auc"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": cumulative_loss / len(train_dataset),
                "validation_pr_auc": validation_pr_auc,
            }
        )
        print(f"epoch={epoch} train_loss={history[-1]['train_loss']:.6f} validation_pr_auc={validation_pr_auc:.6f}")
        if validation_pr_auc > best_pr_auc + 1e-12:
            best_pr_auc = validation_pr_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid validation checkpoint")
    model.load_state_dict(best_state)
    validation_raw_scores = probabilities(model, validation_features, device)
    test_raw_scores = probabilities(model, test_features, device)
    calibration, neural_metrics, threshold, cost_threshold, neural_extras = evaluate_scored_model(
        validation_target=validation_target,
        validation_raw_scores=validation_raw_scores,
        validation_amounts=validation_frame["Amount"].to_numpy(dtype=np.float64),
        test_target=test_target,
        test_raw_scores=test_raw_scores,
        test_amounts=test_frame["Amount"].to_numpy(dtype=np.float64),
        false_positive_review_cost=args.false_positive_review_cost,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=args.seed,
    )
    validation_probabilities = neural_extras["validation_probabilities"]
    test_probabilities = neural_extras["test_probabilities"]

    baselines: dict[str, Any] = {}
    if not args.skip_baselines:
        baselines = run_baselines(
            train_features, train_target, validation_features, validation_target,
            validation_frame["Amount"].to_numpy(dtype=np.float64),
            test_features, test_target, test_frame["Amount"].to_numpy(dtype=np.float64),
            seed=args.seed,
            false_positive_review_cost=args.false_positive_review_cost,
            hgb_iterations=args.baseline_hgb_iterations,
        )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    preprocessor["classification_threshold"] = threshold
    preprocessor["cost_based_threshold"] = cost_threshold
    preprocessor["score_calibration"] = calibration
    preprocessor["model_dimensions"] = list(MODEL_DIMENSIONS)
    preprocessor["training_seed"] = args.seed
    save_preprocessor(preprocessor, args.models_dir / "preprocessor.json")
    torch.save(
        {
            "format": "fraud_pytorch_v1",
            "state_dict": model.cpu().state_dict(),
            "model_dimensions": MODEL_DIMENSIONS,
            "feature_columns": FEATURE_COLUMNS,
            "classification_threshold": threshold,
            "score_calibration": calibration,
        },
        args.models_dir / "fraud_model.pt",
    )
    export_engine_model(model, args.models_dir / "fraud_model.mie")

    predictions = pd.DataFrame(
        {
            "source_index": test_frame.index,
            "actual_class": test_target,
            "raw_engine_score": test_raw_scores,
            "fraud_probability": test_probabilities,
            "predicted_class": (test_probabilities >= threshold).astype(np.int64),
        }
    )
    predictions.to_csv(args.results_dir / "test_predictions.csv", index=False)
    report = {
        "data": {
            "source": project_relative_path(args.data),
            "total_rows": int(len(frame)),
            "fraud_rows": int(frame[TARGET_COLUMN].sum()),
            "genuine_rows": int(len(frame) - frame[TARGET_COLUMN].sum()),
            "split_strategy": args.split_strategy,
            "time_feature": args.time_feature,
            "partitions": {
                "train": partition_summary(train_frame),
                "validation": partition_summary(validation_frame),
                "test": partition_summary(test_frame),
            },
        },
        "model": {
            "name": "fraud_mlp",
            "dimensions": list(MODEL_DIMENSIONS),
            "best_validation_pr_auc_raw_score": best_pr_auc,
            "calibration": calibration,
            "classification_threshold": threshold,
            "cost_based_threshold": cost_threshold,
        },
        "neural_network": {
            **neural_metrics,
            "cost_policy": neural_extras["policy"],
            "bootstrap_confidence_intervals": neural_extras["uncertainty"],
        },
        "baselines": baselines,
        "history": history,
    }
    (args.results_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"neural_network": neural_metrics, "baselines": baselines}, indent=2))
    print(f"Saved model artifacts to {args.models_dir}")
    print(f"Saved evaluation results to {args.results_dir}")


if __name__ == "__main__":
    main()
