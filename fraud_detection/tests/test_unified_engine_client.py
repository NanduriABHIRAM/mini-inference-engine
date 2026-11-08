from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard.inference_client import run_inference, run_reference_workload, runner_path


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def fraud_model(tmp_path: Path) -> Path:
    model = tmp_path / "fixture.mie"
    model.write_text(
        "\n".join(
            [
                "MIE_MLP_V1", "activation sigmoid", "layers 2", "layer 30 2", "weights 60",
                *(["0.01"] * 60), "biases 2", "0.1", "-0.1", "layer 2 1", "weights 2",
                "0.5", "-0.25", "biases 1", "0.05", "",
            ]
        ),
        encoding="utf-8",
    )
    return model


def test_benchmark_and_fraud_use_one_shared_runner_client(fraud_model: Path) -> None:
    if runner_path() is None:
        pytest.skip("Build inference_runner before integration testing")
    benchmark = run_reference_workload(2, "cpu", scheduler_warmups=0, scheduler_trials=1)
    fraud = run_inference(
        backend="auto",
        model_path=fraud_model,
        features=pd.DataFrame(np.full((2, 30), 0.2, dtype=np.float32)),
        scheduler_warmups=0,
        scheduler_trials=1,
    )
    assert benchmark.succeeded, benchmark.error
    assert fraud.succeeded, fraud.error
    assert benchmark.predictions is not None and fraud.predictions is not None
    assert benchmark.predictions["selected_backend"].iloc[0] == "sequential_cpu"
    assert fraud.predictions["requested_backend"].iloc[0] == "auto"
    assert fraud.predictions["selected_backend"].iloc[0] in {"sequential_cpu", "openmp_cpu", "cuda_gpu"}
    assert fraud.predictions["scheduler_selection_latency_ms"].notna().all()
    assert fraud.predictions["fraud_probability"].between(0, 1).all()
