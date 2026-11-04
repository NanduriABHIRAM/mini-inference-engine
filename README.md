# Mini Inference Engine — Starter

The project contains sequential CPU and OpenMP benchmark executables, a CUDA/cuBLAS inference backend (built when CUDA is available), and a Streamlit dashboard.

## Scope and honesty

- Uses deterministic synthetic inputs and deterministic random weights.
- This is a compute workload, **not a trained model or a meaningful classifier**.
- Runtime values are generated only when you run the program on your own computer.
- The CUDA backend is optional and requires a CUDA compiler, CUDA Toolkit, and cuBLAS.

Network: `512 -> 256 -> 256 -> 10`, FP32, ReLU hidden layers, stable Softmax output.

## Dependencies

- C++17 compiler
- CMake 3.16+
- Python 3.10+ for dashboard
- CUDA Toolkit, cuBLAS, and a usable NVIDIA GPU are required to execute the CUDA benchmark.

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

1. Creates deterministic synthetic weights and input batches from the shared model/input generator.
2. Checks output shape, finite values, probability range, Softmax row sums, and repeatability; OpenMP is also compared with the sequential reference.
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

1. Shared deterministic model and input generation is used by sequential CPU, OpenMP, and CUDA runners.
2. CPU and OpenMP benchmarks use the same model, inputs, batch sizes, and CSV field definitions.
3. `inference_cuda` checks numerical agreement with the sequential CPU reference before timing CUDA.
4. The dashboard combines backend measurements and selects the lowest measured median latency for each batch size; this is a profile based decision, so newly measured device data can change the choice.
5. CUDA latency includes transfers and setup. Transfer time is not currently separated from the end-to-end measurement.
6. Optional INT8 quantization remains deferred until FP32 correctness and benchmark coverage are stable.

## Phase 1 correctness checks

Checks output shape, finite values, probabilities in [0,1], each Softmax row sum within `1e-5`, and deterministic repeatability within `1e-7`. These are implementation checks, not model-accuracy tests.

## CUDA backend

CUDA detection is optional. CPU targets remain available when `nvcc` is not installed. With CMake 3.18+, the CUDA Toolkit, and cuBLAS installed, build and run the CUDA comparison runner:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target inference_cuda -j
./build/inference_cuda
```

The runner compares the CUDA backend with the sequential CPU reference using identical deterministic networks and inputs. It checks output agreement, reports median latency and speedup for batch sizes 1, 16, 64, 256, and 1024, and writes measurements to `results/cuda_comparison.csv`. CUDA timings include host/device copies, weight setup, and per-call CUDA/cuBLAS setup, so they represent end-to-end calls rather than kernel-only time. Adjust repetitions with `--warmups N` and `--trials N`.

### WSL

When built in WSL, `inference_cuda` links directly to WSL's CUDA driver bridge at
`/usr/lib/wsl/lib/libcuda.so.1`. This prevents an NVIDIA Linux driver installed
inside the distribution from hiding the GPU exposed by Windows. Reconfigure and
rebuild after pulling this change:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target inference_cuda -j
./build/inference_cuda
```

WSL uses the NVIDIA driver installed on Windows; NVIDIA recommends installing
the CUDA toolkit in WSL without a Linux NVIDIA driver package.

## Suggested Git commits

- `phase1: add sequential CPU MLP baseline`
- `phase1: add correctness checks`
- `phase1: add CSV benchmark output`
- `phase1: add Streamlit dashboard`
