"""Fraud page rendered inside the unified Streamlit application.

This module owns fraud-specific validation, preprocessing, and presentation.
It delegates every prediction to ``dashboard.inference_client.run_inference``;
there is deliberately no second subprocess or backend-selection implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.inference_client import BACKEND_LABELS, BACKEND_OPTIONS, run_inference
from fraud_detection.training.preprocess import (
    FEATURE_COLUMNS,
    InputSchemaError,
    load_preprocessor,
    load_transactions,
    transform_features,
    validate_transaction_frame,
)
from fraud_detection.training.evaluate import calibrate_risk_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "fraud_detection" / "data" / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "fraud_detection" / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.mie"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.json"


@st.cache_data(show_spinner=False)
def _read_transactions(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return load_transactions(path, require_target=False)


@st.cache_data(show_spinner=False)
def _read_preprocessor(path: str, modified_ns: int) -> dict:
    del modified_ns
    return load_preprocessor(path)


def _uploaded_transactions(uploaded_file) -> pd.DataFrame:
    try:
        return validate_transaction_frame(pd.read_csv(uploaded_file), require_target=False)
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise InputSchemaError(f"Could not read uploaded CSV: {exc}") from exc


def _risk_category(probability: pd.Series, low: float, high: float) -> pd.Series:
    return pd.Series(
        np.select([probability < low, probability < high], ["Low", "Medium"], default="High"),
        index=probability.index,
    )


def _render_results(
    result: pd.DataFrame,
    transactions: pd.DataFrame,
    preprocessor: dict,
    threshold: float,
    low: float,
    high: float,
    terminal: str,
) -> None:
    display = transactions.reset_index(names="source_row").copy()
    raw_scores = pd.to_numeric(result["fraud_probability"], errors="raise").to_numpy(dtype=np.float64)
    display["fraud_probability"] = calibrate_risk_scores(raw_scores, preprocessor.get("score_calibration"))
    display["raw_engine_score"] = raw_scores
    display["prediction"] = np.where(display["fraud_probability"] >= threshold, "Fraud", "Genuine")
    display["risk"] = _risk_category(display["fraud_probability"], low, high)

    first = result.iloc[0]
    total = len(display)
    fraud_count = int((display["prediction"] == "Fraud").sum())
    columns = st.columns(6)
    columns[0].metric("Backend used", BACKEND_LABELS.get(str(first["selected_backend"]), str(first["selected_backend"])))
    columns[1].metric("Transactions", total)
    columns[2].metric("Predicted genuine", total - fraud_count)
    columns[3].metric("Predicted fraud", fraud_count)
    columns[4].metric("Fraud percentage", f"{100 * fraud_count / total:.2f}%")
    columns[5].metric("Inference latency", f"{float(first['latency_ms']):.3f} ms")
    total_latency = float(first["total_request_latency_ms"])
    st.caption(
        f"Throughput: {1000 * total / float(first['latency_ms']):.1f} transactions/s. "
        f"Total request latency: {total_latency:.3f} ms."
    )
    if pd.notna(first.get("scheduler_selection_latency_ms")):
        st.caption(f"Adaptive scheduler selection: {float(first['scheduler_selection_latency_ms']):.3f} ms.")

    timing_columns = {
        "scheduler_cpu_median_ms": "Sequential CPU",
        "scheduler_openmp_median_ms": "OpenMP CPU",
        "scheduler_gpu_median_ms": "CUDA GPU",
    }
    timings = [
        f"{label} {float(first[column]):.3f} ms"
        for column, label in timing_columns.items()
        if column in result and pd.notna(first[column])
    ]
    if timings:
        st.caption("Scheduler measurements: " + ", ".join(timings))

    table_columns = ["source_row", "fraud_probability", "raw_engine_score", "prediction", "risk"]
    if "Class" in display:
        table_columns.insert(1, "Class")
    st.subheader("Predictions")
    st.dataframe(display.loc[:, table_columns].sort_values("fraud_probability", ascending=False),
                 use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        chart = px.histogram(display, x="fraud_probability", color="prediction", nbins=30,
                             title="Fraud probability distribution")
        chart.add_vline(x=threshold, line_dash="dash", annotation_text="Decision threshold")
        st.plotly_chart(chart, use_container_width=True)
    with right:
        counts = display["risk"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
        chart = px.bar(x=counts.index, y=counts.values, color=counts.index, title="Risk category distribution",
                       labels={"x": "Risk category", "y": "Transactions"},
                       color_discrete_map={"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"})
        chart.update_layout(showlegend=False)
        st.plotly_chart(chart, use_container_width=True)
    st.download_button("Download prediction results", display.to_csv(index=False).encode("utf-8"),
                       "fraud_predictions.csv", "text/csv")
    with st.expander("Engine details"):
        st.code(terminal or "No runner output.")


def render() -> None:
    """Render fraud detection in the single application, using the shared engine client."""

    st.header("Credit Card Fraud Detection")
    st.caption("Saved preprocessing and the trained sigmoid model are executed only by the shared C++ engine.")
    if not PREPROCESSOR_PATH.is_file() or not MODEL_PATH.is_file():
        st.error("No trained fraud model was found.")
        st.code("python -m fraud_detection.training.train")
        return
    try:
        preprocessor = _read_preprocessor(str(PREPROCESSOR_PATH), PREPROCESSOR_PATH.stat().st_mtime_ns)
    except InputSchemaError as exc:
        st.error(str(exc))
        return

    source = st.radio("Transaction source", ["Project dataset", "Upload CSV"], horizontal=True)
    transactions: pd.DataFrame | None = None
    if source == "Project dataset":
        if not DATASET_PATH.is_file():
            st.warning(f"Dataset not found: {DATASET_PATH.relative_to(PROJECT_ROOT)}")
        else:
            try:
                transactions = _read_transactions(str(DATASET_PATH), DATASET_PATH.stat().st_mtime_ns)
            except InputSchemaError as exc:
                st.error(str(exc))
    else:
        upload = st.file_uploader("Upload Kaggle-format transactions", type=["csv"])
        st.caption("Required columns: Time, V1–V28, Amount. Class is optional.")
        if upload is not None:
            try:
                transactions = _uploaded_transactions(upload)
            except InputSchemaError as exc:
                st.error(str(exc))
    if transactions is None:
        return

    st.caption(f"Loaded {len(transactions):,} transactions. Feature order is fixed by the saved preprocessing artifact.")
    st.dataframe(transactions.head(10), use_container_width=True, hide_index=True)
    mode = st.radio("Inference mode", ["One transaction", "Batch"], horizontal=True)
    if mode == "One transaction":
        row = int(st.number_input("Transaction row", 0, len(transactions) - 1, 0))
        selected = transactions.iloc[[row]].copy()
    else:
        start = int(st.number_input("Batch start row", 0, len(transactions) - 1, 0))
        maximum = len(transactions) - start
        size = int(st.number_input("Batch size", 1, maximum, min(128, maximum)))
        selected = transactions.iloc[start:start + size].copy()

    saved_threshold = float(preprocessor.get("classification_threshold", 0.5))
    first, second, third, fourth = st.columns(4)
    with first:
        backend = st.selectbox("Backend", BACKEND_OPTIONS, format_func=BACKEND_LABELS.get, key="fraud_backend")
    with second:
        threshold = float(st.number_input("Fraud threshold", 0.0, 1.0, saved_threshold, 0.01))
    with third:
        low = float(st.number_input("Low-risk upper bound", 0.0, 1.0, 0.30, 0.01))
    with fourth:
        high = float(st.number_input("High-risk lower bound", 0.0, 1.0, 0.70, 0.01))

    if st.button(f"Run fraud inference for {len(selected):,} transaction(s)", type="primary"):
        if low >= high:
            st.error("The low-risk bound must be less than the high-risk bound.")
        else:
            try:
                transformed = transform_features(selected, preprocessor)
            except InputSchemaError as exc:
                st.error(str(exc))
                return
            with st.spinner("Running the shared C++ inference engine..."):
                run = run_inference(
                    backend=backend,
                    model_path=MODEL_PATH,
                    features=pd.DataFrame(transformed, columns=FEATURE_COLUMNS),
                    # Classification is performed after calibration below;
                    # the C++ threshold only labels its raw diagnostic column.
                    decision_threshold=0.5,
                )
            if not run.succeeded:
                st.error(run.error)
                if run.terminal_output:
                    st.code(run.terminal_output)
            else:
                st.session_state["fraud_results"] = (
                    run.predictions, selected, preprocessor, threshold, low, high, run.terminal_output
                )

    saved = st.session_state.get("fraud_results")
    if saved:
        _render_results(*saved)
