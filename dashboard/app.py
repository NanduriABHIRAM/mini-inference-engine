"""Unified Streamlit interface for benchmark and fraud inference workflows."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.inference_client import (  # noqa: E402
    BACKEND_LABELS,
    BACKEND_OPTIONS,
    build_runner,
    run_reference_workload,
    runner_path,
)
from fraud_detection.dashboard.fraud_page import render as render_fraud_page  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results"


def _standard_profile(path: Path, rank: int) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file() or not path.stat().st_size:
        return None, None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # surfaced to the dashboard, not silently discarded
        return None, f"Could not read {path.name}: {exc}"
    required = {"backend", "batch_size", "median_latency_ms"}
    if not required.issubset(frame):
        return None, f"{path.name} does not contain a compatible benchmark profile."
    frame = frame.copy()
    frame["source_file"] = path.name
    frame["source_rank"] = rank
    frame["record_order"] = range(len(frame))
    return frame, None


def _cuda_profile(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.is_file() or not path.stat().st_size:
        return None, None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        return None, f"Could not read {path.name}: {exc}"
    if not {"batch_size", "gpu_median_ms"}.issubset(frame):
        return None, f"{path.name} does not contain a compatible CUDA benchmark profile."
    profile = pd.DataFrame({
        "backend": "cuda_gpu",
        "batch_size": frame["batch_size"],
        "median_latency_ms": frame["gpu_median_ms"],
        "source_file": path.name,
        "source_rank": 3,
        "record_order": range(len(frame)),
    })
    return profile, None


def _profiles() -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    notices: list[str] = []
    for path, rank, reader in (
        (RESULTS_DIR / "benchmark.csv", 1, _standard_profile),
        (RESULTS_DIR / "openmp.csv", 2, _standard_profile),
        (RESULTS_DIR / "cuda_comparison.csv", 3, _cuda_profile),
    ):
        frame, notice = reader(path, rank) if reader is _standard_profile else reader(path)
        if frame is not None:
            frames.append(frame)
        if notice:
            notices.append(notice)
    if not frames:
        return pd.DataFrame(), notices
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "status" in combined:
        combined = combined[combined["status"].fillna("measured").eq("measured")]
    for column in ("batch_size", "median_latency_ms"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    combined = combined.dropna(subset=["backend", "batch_size", "median_latency_ms"])
    combined = combined[(combined["batch_size"] > 0) & (combined["median_latency_ms"] > 0)].copy()
    if combined.empty:
        return combined, notices
    combined["batch_size"] = combined["batch_size"].astype(int)
    combined = combined.sort_values(["backend", "batch_size", "source_rank", "record_order"], kind="stable")
    combined = combined.drop_duplicates(["backend", "batch_size"], keep="last")
    combined["throughput_samples_per_sec"] = 1000 * combined["batch_size"] / combined["median_latency_ms"]
    baseline = combined[combined["backend"] == "sequential_cpu"].set_index("batch_size")["median_latency_ms"]
    combined["speedup_vs_sequential"] = combined.apply(
        lambda row: baseline.get(row["batch_size"], float("nan")) / row["median_latency_ms"], axis=1
    )
    combined["backend_label"] = combined["backend"].map(BACKEND_LABELS).fillna(combined["backend"])
    return combined.sort_values(["batch_size", "backend"]).reset_index(drop=True), notices


def _render_runner_status() -> None:
    left, right = st.columns([3, 1])
    with left:
        executable = runner_path()
        if executable is None:
            st.warning("Shared C++ runner is not built.")
        else:
            st.success(f"Shared engine: {executable.relative_to(PROJECT_ROOT)}")
    with right:
        if st.button("Build engine", use_container_width=True):
            with st.spinner("Configuring and building inference_runner..."):
                built, output = build_runner()
            if built:
                st.success("Engine built.")
            else:
                st.error("Engine build failed.")
            with st.expander("Build output"):
                st.code(output or "No output returned.")


def _render_live_benchmark() -> None:
    st.subheader("Run benchmark experiment")
    st.caption("This invokes the same C++ runner and backend selector used by fraud predictions.")
    left, right = st.columns(2)
    with left:
        batch_size = int(st.selectbox("Batch size", [1, 16, 64, 256, 1024], index=2))
    with right:
        backend = st.selectbox("Backend", BACKEND_OPTIONS, format_func=BACKEND_LABELS.get, key="benchmark_backend")
    if st.button("Run selected experiment", type="primary"):
        with st.spinner("Running reference workload through the shared engine..."):
            run = run_reference_workload(batch_size, backend)
        if not run.succeeded:
            st.error(run.error)
            if run.terminal_output:
                st.code(run.terminal_output)
        else:
            result = run.predictions
            first = result.iloc[0]
            columns = st.columns(5)
            columns[0].metric("Backend used", BACKEND_LABELS.get(str(first["selected_backend"]), str(first["selected_backend"])))
            columns[1].metric("Inference latency", f"{float(first['latency_ms']):.3f} ms")
            columns[2].metric("Throughput", f"{1000 * batch_size / float(first['latency_ms']):.1f} samples/s")
            columns[3].metric("Total request", f"{float(first['total_request_latency_ms']):.3f} ms")
            selection = first.get("scheduler_selection_latency_ms")
            columns[4].metric("Scheduler selection", "N/A" if pd.isna(selection) else f"{float(selection):.3f} ms")
            with st.expander("Engine details"):
                st.code(run.terminal_output)


def _render_saved_profiles() -> None:
    st.subheader("Measured benchmark profiles")
    profiles, notices = _profiles()
    for notice in notices:
        st.warning(notice)
    if profiles.empty:
        st.info("No saved profile found. Run inference_cpu, inference_openmp, and optionally inference_cuda to create one.")
        return
    best = profiles.loc[profiles.groupby("batch_size")["median_latency_ms"].idxmin(),
                        ["batch_size", "backend_label", "median_latency_ms"]].rename(
                            columns={"backend_label": "selected_backend", "median_latency_ms": "selected_latency_ms"})
    one, two, three = st.columns(3)
    one.metric("Recorded profiles", len(profiles))
    two.metric("Backends", profiles["backend"].nunique())
    three.metric("Batch sizes", profiles["batch_size"].nunique())
    for title, metric, axis in (
        ("Latency", "median_latency_ms", "Median latency (ms)"),
        ("Throughput", "throughput_samples_per_sec", "Samples/second"),
        ("Speedup vs sequential CPU", "speedup_vs_sequential", "Speedup"),
    ):
        st.markdown(f"#### {title}")
        chart = px.line(profiles, x="batch_size", y=metric, color="backend_label", markers=True, log_x=True)
        chart.update_layout(xaxis_title="Batch size (log scale)", yaxis_title=axis)
        st.plotly_chart(chart, use_container_width=True)
    st.markdown("#### Adaptive scheduler decisions from measured profiles")
    st.dataframe(best, use_container_width=True, hide_index=True)
    st.dataframe(profiles.drop(columns=["source_rank", "record_order"]), use_container_width=True, hide_index=True)
    st.download_button("Download benchmark CSV", profiles.to_csv(index=False).encode("utf-8"),
                       "inference_results.csv", "text/csv")


def render_benchmark_page() -> None:
    st.header("Benchmark")
    _render_live_benchmark()
    _render_saved_profiles()


def main() -> None:
    st.set_page_config(page_title="Mini Inference Engine", layout="wide")
    st.title("Mini Inference Engine")
    st.caption("One C++ engine, one backend selector, two workloads.")
    _render_runner_status()
    page = st.sidebar.radio("Application", ["Benchmark", "Fraud Detection"])
    if page == "Benchmark":
        render_benchmark_page()
    else:
        render_fraud_page()


if __name__ == "__main__":
    main()
