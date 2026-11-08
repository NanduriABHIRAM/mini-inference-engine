# Mini Inference Engine

One inference engine serves two workloads: a deterministic MLP benchmark and
credit-card fraud detection. Both use the same C++ `inference_runner` and the
same four backend choices: sequential CPU, OpenMP CPU, CUDA/cuBLAS GPU, and
adaptive AUTO selection.

## Architecture

```text
Streamlit UI (dashboard/app.py)
├── Benchmark page ──────── shared Python engine client ─┐
└── Fraud Detection page ── shared Python engine client ─┼─ inference_runner
                                                         ├─ sequential CPU
                                                         ├─ OpenMP CPU
                                                         └─ CUDA/cuBLAS GPU
```

The fraud model is trained in Python, exported in portable `MIE_MLP_V1` form,
then loaded by the same C++ dense-layer, activation, validation, and scheduler
path as the benchmark. Fraud preprocessing, feature order, calibration, and
threshold metadata are saved beside the model.

## Build

Requirements: C++17, CMake 3.16+, Python 3.10+. CUDA is optional. OpenMP is
also optional; when a compiler does not support it, CMake still builds the
project and labels the OpenMP benchmark executable as a sequential fallback.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

CUDA architecture is configurable rather than hard-coded:

```bash
cmake -S . -B build -DMIE_CUDA_ARCHITECTURES="75;86;89"
```

Run benchmark executables directly when desired:

```bash
./build/inference_cpu --csv results/benchmark.csv
./build/inference_openmp --csv results/openmp.csv
./build/inference_cuda --csv results/cuda_comparison.csv  # when CUDA is usable
```

## Unified UI

Install the application dependencies, build the shared runner, and launch the
single UI:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dashboard.txt
cmake --build build --target inference_runner --parallel
streamlit run dashboard/app.py
```

Use the sidebar to switch between **Benchmark** and **Fraud Detection**. The
backend selector is shared. Every result shows the actual backend selected by
the C++ runner. AUTO profiles available CPU/OpenMP/CUDA candidates on the
current model and batch, verifies numerical agreement with sequential CPU, and
selects the fastest valid candidate.

For AUTO, result CSVs distinguish:

- `latency_ms`: final selected-backend inference call.
- `scheduler_selection_latency_ms`: adaptive profiling and selection.
- `total_request_latency_ms`: selection plus final inference.

CUDA requests fall back through the same measured AUTO path when CUDA is not
available; the requested and actual backends are both recorded.

## Fraud training

Download Kaggle's [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place `creditcard.csv` in `fraud_detection/data/`. This dataset is not
included in the repository.

```bash
source .venv/bin/activate
python -m fraud_detection.training.train
```

The default training workflow uses a chronological 70/15/15 split, fits scaling
on training data only, represents `Time` as hour-of-day, uses weighted binary
loss, early-stops on validation PR-AUC, calibrates raw neural-network scores
with validation-only Platt scaling, and tunes both F1 and transparent
amount/review-cost policies on validation. It reports held-out precision,
recall, F1, PR-AUC, ROC-AUC, confusion matrix, bootstrap confidence intervals,
and Logistic Regression plus HistGradientBoosting baselines in
`fraud_detection/results/evaluation.json`.

Useful options:

```bash
python -m fraud_detection.training.train --split-strategy stratified_random
python -m fraud_detection.training.train --false-positive-review-cost 10
python -m fraud_detection.training.train --skip-baselines
```

Generated datasets, models, results, builds, caches, and source ZIPs are
ignored by Git. To create a clean source archive:

```bash
python scripts/package_source.py --output /tmp/mini-inference-engine-source.zip
```

## Tests

After dependencies are installed and the runner is built:

```bash
pytest -q fraud_detection/tests
```

The CUDA parity test automatically skips only if no usable CUDA device exists.
The model-artifact parity test automatically skips until the Kaggle dataset has
been trained locally.

## Current limitations

- CUDA is optional and unavailable environments use verified CPU/OpenMP
  fallback; no GPU performance claim is fabricated.
- The runner is process-per-request. CUDA context setup, weight uploads, and
  cuBLAS-handle construction remain inside a backend call, so small batches are
  intentionally measured end-to-end and may favor CPU. A future long-lived
  service can retain GPU-resident model state for high-throughput scoring.
- The cost policy uses a configurable fixed false-positive review cost; replace
  that assumption with your organisation's economics before deployment.
