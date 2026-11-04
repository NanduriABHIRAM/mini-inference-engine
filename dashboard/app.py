from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Mini Inference Engine", layout="wide")
st.title("Mini Inference Engine — Benchmark Dashboard")
st.caption("Measured backend profiles and per batch scheduler selections.")

results_dir = Path(__file__).resolve().parents[1] / "results"
frames = []
for filename in ("benchmark.csv", "openmp.csv", "cuda_comparison.csv"):
    path = results_dir / filename
    if not path.exists() or not path.stat().st_size:
        continue
    try:
        part = pd.read_csv(path)
        if {"backend", "batch_size", "median_latency_ms"}.issubset(part.columns):
            frames.append(part)
        elif {"batch_size", "cpu_median_ms", "gpu_median_ms"}.issubset(part.columns):
            for backend, column in (("sequential_cpu", "cpu_median_ms"),
                                    ("cuda_gpu", "gpu_median_ms")):
                profile = part[["batch_size", column]].rename(
                    columns={column: "median_latency_ms"})
                profile["backend"] = backend
                profile["status"] = "measured"
                frames.append(profile)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")

if not frames:
    st.warning("No benchmark data found. Build and run a CPU benchmark first.")
    st.code("cmake -S . -B build\ncmake --build build -j\n./build/inference_cpu\n./build/inference_openmp")
    st.stop()

df = pd.concat(frames, ignore_index=True, sort=False)
if "status" in df.columns:
    df = df[df["status"] == "measured"].copy()
for column in ("batch_size", "median_latency_ms"):
    df[column] = pd.to_numeric(df[column], errors="coerce")
df = df.dropna(subset=["batch_size", "median_latency_ms"])
df = df[df["median_latency_ms"] > 0].drop_duplicates(
    subset=["backend", "batch_size"], keep="last")
df["throughput_samples_per_sec"] = 1000 * df["batch_size"] / df["median_latency_ms"]
df["transfer_overhead_ms"] = "N/A (CPU)"
df.loc[df["backend"] == "cuda_gpu", "transfer_overhead_ms"] = "Included in GPU latency; not split out"

seq = df[df["backend"] == "sequential_cpu"].set_index("batch_size")["median_latency_ms"]
df["speedup_vs_sequential"] = df.apply(
    lambda row: seq.get(row["batch_size"], float("nan")) / row["median_latency_ms"], axis=1)
best = df.loc[df.groupby("batch_size")["median_latency_ms"].idxmin(),
              ["batch_size", "backend", "median_latency_ms"]].rename(
                  columns={"backend": "selected_backend",
                           "median_latency_ms": "selected_latency_ms"})
df = df.merge(best, on="batch_size", how="left")

first, second, third = st.columns(3)
first.metric("Measured profiles", len(df))
second.metric("Backends", df["backend"].nunique())
third.metric("Batch sizes", df["batch_size"].nunique())

st.subheader("Latency")
fig = px.line(df, x="batch_size", y="median_latency_ms", color="backend",
              markers=True, log_x=True)
fig.update_layout(xaxis_title="Batch size (log scale)", yaxis_title="Median latency (ms)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Throughput")
fig2 = px.line(df, x="batch_size", y="throughput_samples_per_sec", color="backend",
               markers=True, log_x=True)
fig2.update_layout(xaxis_title="Batch size (log scale)", yaxis_title="Samples/second")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Adaptive scheduler decisions")
st.dataframe(best, use_container_width=True)
st.caption("The scheduler selects the lowest measured median latency for each batch size. CUDA call latency includes transfers and setup; transfer overhead is not separately instrumented yet.")

st.subheader("Measured data")
st.dataframe(df, use_container_width=True)
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                   "inference_results.csv", "text/csv")
