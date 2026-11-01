from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Mini Inference Engine", layout="wide")
st.title("Mini Inference Engine — Benchmark Dashboard")
st.caption("Phase 1: Sequential CPU baseline. Only measured CSV rows are displayed.")

csv_path = Path(__file__).resolve().parents[1] / "results" / "benchmark.csv"
if not csv_path.exists():
    st.warning("No benchmark CSV yet. Build and run the C++ program first.")
    st.code("cmake -S . -B build\ncmake --build build\n./build/inference_cpu")
    st.stop()

try:
    df = pd.read_csv(csv_path)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

needed = {"backend", "batch_size", "median_latency_ms", "throughput_samples_per_sec", "status"}
if not needed.issubset(df.columns):
    st.error("CSV columns do not match. Re-run the supplied benchmark executable.")
    st.stop()

df = df[df["status"] == "measured"].copy()
if df.empty:
    st.info("No measured rows yet. Run the benchmark executable.")
    st.stop()

for col in ["batch_size", "median_latency_ms", "throughput_samples_per_sec"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["batch_size", "median_latency_ms", "throughput_samples_per_sec"])

a, b, c = st.columns(3)
a.metric("Measured rows", len(df))
b.metric("Backends measured", df["backend"].nunique())
c.metric("Batch sizes", df["batch_size"].nunique())

st.subheader("Median latency")
fig = px.line(df, x="batch_size", y="median_latency_ms", color="backend", markers=True,
              log_x=True, title="Measured median inference latency")
fig.update_layout(xaxis_title="Batch size (log scale)", yaxis_title="Latency (ms)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Throughput")
fig2 = px.line(df, x="batch_size", y="throughput_samples_per_sec", color="backend",
               markers=True, log_x=True, title="Measured inference throughput")
fig2.update_layout(xaxis_title="Batch size (log scale)", yaxis_title="Samples/second")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Measured data")
st.dataframe(df, use_container_width=True)
st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                   "inference_results.csv", "text/csv")
st.caption("No missing GPU results are estimated or filled in.")
