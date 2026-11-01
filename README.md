# Mini Inference Engine — Starter

Phase 1 contains a sequential CPU MLP inference baseline, basic correctness checks, a CSV benchmark writer, and a Streamlit dashboard.

## Scope and honesty

- Uses deterministic synthetic inputs and deterministic random weights.
- This is a compute workload, **not a trained model or a meaningful classifier**.
- Runtime values are generated only when you run the program on your own computer.
- No OpenMP or CUDA code is included yet. Add those after Phase 1 builds and passes.

Network: `512 -> 256 -> 256 -> 10`, FP32, ReLU hidden layers, stable Softmax output.

## Dependencies

- C++17 compiler
- CMake 3.16+
- Python 3.10+ for dashboard
- CUDA Toolkit is needed in a later phase, not Phase 1.

### Ubuntu

```bash
sudo apt update
sudo apt install -y build-essential cmake python3 python3-venv python3-pip
```

### Build/run on Ubuntu or Linux

Run from the project root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/inference_cpu
```

Custom trial count:

```bash
./build/inference_cpu --warmups 3 --trials 7 --csv results/benchmark.csv
```

### Windows

Install Visual Studio Build Tools with the **Desktop development with C++** workload and CMake. In a Developer PowerShell:

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\inference_cpu.exe
```

If the executable is elsewhere, locate `inference_cpu.exe` inside `build`.

## What the program does

1. Creates deterministic synthetic weights and input batches.
2. Checks output shape, finite values, probability range, Softmax row sums, and repeatability.
3. Runs warm-ups, then repeated timed inference.
4. Reports median latency and throughput for batch sizes 1, 16, 64, 256, 1024.
5. Writes actual run results to `results/benchmark.csv`.

The exact timings depend on your machine. This README contains no fabricated timings.

## Dashboard

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Run the C++ benchmark before starting the dashboard.

## End-semester implementation roadmap

1. **Phase 1 — Sequential CPU baseline** (included).
2. **Phase 2 — OpenMP backend:** parallelize independent batch rows and compare output with CPU reference.
3. **Phase 3 — CUDA/cuBLAS:** use cuBLAS SGEMM for dense layers; CUDA kernels for bias/activation; verify against CPU.
4. **Phase 4 — Benchmark protocol:** include transfer timing; report latency, throughput, speedup and efficiency.
5. **Phase 5 — Adaptive scheduler:** use measured profiles to select a backend for each workload. Do not hardcode unmeasured crossover claims.
6. **Phase 6 — Dashboard:** compare backends, show scheduler decision and export plots.
7. **Optional INT8 extension:** only after FP32 implementations are correct and benchmarked.

## Phase 1 correctness checks

Checks output shape, finite values, probabilities in [0,1], each Softmax row sum within `1e-5`, and deterministic repeatability within `1e-7`. These are implementation checks, not model-accuracy tests. CPU/GPU agreement will be added when multiple backends exist.

## Suggested Git commits

- `phase1: add sequential CPU MLP baseline`
- `phase1: add correctness checks`
- `phase1: add CSV benchmark output`
- `phase1: add Streamlit dashboard`
