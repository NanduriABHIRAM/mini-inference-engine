"""One Python client for the compiled Mini Inference Engine runner.

Both pages in the Streamlit application use this module.  It deliberately does
not implement inference in Python: it serialises inputs, invokes the existing
``inference_runner`` executable, and returns its measured results.  This keeps
the benchmark and fraud workflows on the same CPU, OpenMP, CUDA, and adaptive
AUTO scheduler paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = PROJECT_ROOT / "build"
BACKEND_OPTIONS = ("auto", "cpu", "openmp", "gpu")
BACKEND_LABELS = {
    "auto": "Adaptive (fastest verified backend)",
    "cpu": "Sequential CPU",
    "openmp": "OpenMP CPU",
    "gpu": "CUDA GPU",
    "sequential_cpu": "Sequential CPU",
    "openmp_cpu": "OpenMP CPU",
    "cuda_gpu": "CUDA GPU",
}


@dataclass(frozen=True)
class EngineRun:
    """Result of one request made through the C++ inference runner."""

    predictions: pd.DataFrame | None
    terminal_output: str
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.predictions is not None and self.predictions.size > 0 and not self.error


def runner_path() -> Path | None:
    """Find the CMake-built runner on Unix or Windows."""

    names = ("inference_runner.exe", "inference_runner") if os.name == "nt" else ("inference_runner",)
    candidates = [BUILD_DIR / name for name in names]
    candidates.extend(BUILD_DIR / "Release" / name for name in names)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def build_runner() -> tuple[bool, str]:
    """Configure and build the single runner used by both application pages."""

    try:
        configured = subprocess.run(
            ["cmake", "-S", str(PROJECT_ROOT), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if configured.returncode:
            return False, configured.stdout + configured.stderr
        built = subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--target", "inference_runner", "--parallel"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        return built.returncode == 0, built.stdout + built.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def _as_feature_frame(features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> pd.DataFrame:
    frame = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("Inference requires at least one non-empty feature row.")
    try:
        values = frame.to_numpy(dtype=np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Inference features must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError("Inference features must be finite.")
    return pd.DataFrame(values, columns=frame.columns)


def run_inference(
    *,
    backend: str,
    model_path: Path | None = None,
    features: pd.DataFrame | np.ndarray | Sequence[Sequence[float]] | None = None,
    samples: int | None = None,
    decision_threshold: float = 0.5,
    scheduler_warmups: int | None = None,
    scheduler_trials: int | None = None,
    timeout_seconds: int = 180,
) -> EngineRun:
    """Run the existing C++ engine and return its CSV result.

    A custom model requires explicit feature rows.  Omitting ``model_path``
    and providing ``samples`` runs the deterministic reference workload used
    by the benchmark page.  ``backend`` is passed through unchanged to the
    runner, including ``auto``; the C++ scheduler remains the sole selector.
    """

    if backend not in BACKEND_OPTIONS:
        return EngineRun(None, "", f"Unsupported backend: {backend}")
    executable = runner_path()
    if executable is None:
        return EngineRun(None, "", "The C++ inference runner is not built yet. Use Build engine first.")
    if not np.isfinite(decision_threshold) or not 0.0 <= decision_threshold <= 1.0:
        return EngineRun(None, "", "The decision threshold must be between 0 and 1.")
    if model_path is not None and not model_path.is_file():
        return EngineRun(None, "", f"Model file is missing: {model_path}")
    if (features is None) == (samples is None):
        return EngineRun(None, "", "Specify exactly one of features or samples.")
    if samples is not None and samples <= 0:
        return EngineRun(None, "", "Sample count must be greater than zero.")

    try:
        with tempfile.TemporaryDirectory(prefix="mie-run-") as temporary_directory:
            work_dir = Path(temporary_directory)
            output_path = work_dir / "predictions.csv"
            command = [str(executable), "--backend", backend, "--output", str(output_path)]
            if model_path is not None:
                command.extend(["--model", str(model_path)])
            if features is not None:
                input_path = work_dir / "features.csv"
                _as_feature_frame(features).to_csv(input_path, index=False, float_format="%.9g")
                command.extend(["--input", str(input_path)])
            else:
                command.extend(["--samples", str(samples)])
            command.extend(["--decision-threshold", str(decision_threshold)])
            if scheduler_warmups is not None:
                command.extend(["--scheduler-warmups", str(scheduler_warmups)])
            if scheduler_trials is not None:
                command.extend(["--scheduler-trials", str(scheduler_trials)])

            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            terminal = completed.stdout + completed.stderr
            if completed.returncode:
                return EngineRun(None, terminal, "The selected inference backend did not complete successfully.")
            if not output_path.is_file():
                return EngineRun(None, terminal, "The engine completed without writing prediction results.")
            predictions = pd.read_csv(output_path)
            if predictions.empty:
                return EngineRun(None, terminal, "The engine produced an empty result.")
            return EngineRun(predictions, terminal)
    except (OSError, subprocess.TimeoutExpired, pd.errors.ParserError, ValueError) as exc:
        return EngineRun(None, "", str(exc))


def run_reference_workload(
    batch_size: int,
    backend: str,
    *,
    scheduler_warmups: int = 1,
    scheduler_trials: int = 5,
) -> EngineRun:
    """Run the benchmark's deterministic reference MLP through the shared runner."""

    return run_inference(
        backend=backend,
        samples=batch_size,
        scheduler_warmups=scheduler_warmups,
        scheduler_trials=scheduler_trials,
    )
