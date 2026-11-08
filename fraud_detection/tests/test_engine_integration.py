from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_detection.training.model import FraudMLP
from fraud_detection.training.preprocess import load_preprocessor, load_transactions, transform_features


ROOT = Path(__file__).resolve().parents[2]


def runner_path() -> Path | None:
    candidates = [ROOT / "build" / "inference_runner", ROOT / "build" / "Release" / "inference_runner.exe"]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


@pytest.fixture()
def fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "fixture.mie"
    # 30 -> 2 -> 1 ReLU/sigmoid model with deterministic FP32 weights.
    weights_1 = "\n".join(["0.01"] * 60)
    model.write_text(
        "\n".join(
            [
                "MIE_MLP_V1",
                "activation sigmoid",
                "layers 2",
                "layer 30 2",
                "weights 60",
                weights_1,
                "biases 2",
                "0.1",
                "-0.1",
                "layer 2 1",
                "weights 2",
                "0.5",
                "-0.25",
                "biases 1",
                "0.05",
                "",
            ]
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "features.csv"
    pd.DataFrame(np.full((3, 30), 0.2, dtype=np.float32)).to_csv(input_path, index=False)
    return model, input_path


def invoke(runner: Path, model: Path, input_path: Path, output: Path, backend: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(runner),
            "--backend",
            backend,
            "--model",
            str(model),
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--scheduler-warmups",
            "0",
            "--scheduler-trials",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cpu_openmp_and_auto_fraud_inference(fixture_files: tuple[Path, Path], tmp_path: Path) -> None:
    runner = runner_path()
    if runner is None:
        pytest.skip("Build inference_runner before integration testing")
    model, input_path = fixture_files
    cpu_path, openmp_path, auto_path = (tmp_path / "cpu.csv", tmp_path / "openmp.csv", tmp_path / "auto.csv")

    cpu = invoke(runner, model, input_path, cpu_path, "cpu")
    openmp = invoke(runner, model, input_path, openmp_path, "openmp")
    automatic = invoke(runner, model, input_path, auto_path, "auto")
    assert cpu.returncode == 0, cpu.stderr
    assert openmp.returncode == 0, openmp.stderr
    assert automatic.returncode == 0, automatic.stderr

    cpu_result = pd.read_csv(cpu_path)
    openmp_result = pd.read_csv(openmp_path)
    auto_result = pd.read_csv(auto_path)
    assert cpu_result["output_activation"].eq("sigmoid").all()
    assert cpu_result["fraud_probability"].between(0, 1).all()
    np.testing.assert_allclose(cpu_result["fraud_probability"], openmp_result["fraud_probability"], atol=1e-5)
    assert auto_result["selected_backend"].iloc[0] in {"sequential_cpu", "openmp_cpu", "cuda_gpu"}
    assert auto_result["scheduler_cpu_median_ms"].notna().all()
    assert auto_result["scheduler_selection_latency_ms"].notna().all()
    assert (auto_result["total_request_latency_ms"] >= auto_result["latency_ms"]).all()


def test_cuda_fraud_consistency_when_a_device_is_usable(
    fixture_files: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = runner_path()
    if runner is None:
        pytest.skip("Build inference_runner before integration testing")
    model, input_path = fixture_files
    cpu_path, gpu_path = tmp_path / "cpu.csv", tmp_path / "gpu.csv"
    assert invoke(runner, model, input_path, cpu_path, "cpu").returncode == 0
    gpu = invoke(runner, model, input_path, gpu_path, "gpu")
    assert gpu.returncode == 0, gpu.stderr
    cpu = pd.read_csv(cpu_path)
    gpu_result = pd.read_csv(gpu_path)
    if gpu_result["selected_backend"].iloc[0] != "cuda_gpu":
        pytest.skip("CUDA is unavailable; inference_runner selected a measured CPU/OpenMP fallback")
    np.testing.assert_allclose(cpu["fraud_probability"], gpu_result["fraud_probability"], atol=1e-4)


def test_gpu_request_falls_back_when_cuda_is_unavailable(
    fixture_files: tuple[Path, Path], tmp_path: Path
) -> None:
    runner = runner_path()
    if runner is None:
        pytest.skip("Build inference_runner before integration testing")
    model, input_path = fixture_files
    output_path = tmp_path / "requested_gpu.csv"
    completed = invoke(runner, model, input_path, output_path, "gpu")
    assert completed.returncode == 0, completed.stderr
    result = pd.read_csv(output_path)
    assert result["requested_backend"].eq("gpu").all()
    assert result["selected_backend"].iloc[0] in {"sequential_cpu", "openmp_cpu", "cuda_gpu"}
    assert result["fraud_probability"].between(0, 1).all()


def test_exported_pytorch_model_matches_cpp_when_artifacts_exist(tmp_path: Path) -> None:
    runner = runner_path()
    model_path = ROOT / "fraud_detection" / "models" / "fraud_model.mie"
    torch_path = ROOT / "fraud_detection" / "models" / "fraud_model.pt"
    preprocessor_path = ROOT / "fraud_detection" / "models" / "preprocessor.json"
    data_path = ROOT / "fraud_detection" / "data" / "creditcard.csv"
    if runner is None or not all(path.is_file() for path in [model_path, torch_path, preprocessor_path, data_path]):
        pytest.skip("Train the fraud model and build inference_runner before this parity test")

    import torch

    artifact = torch.load(torch_path, map_location="cpu", weights_only=False)
    model = FraudMLP()
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    raw = load_transactions(data_path, require_target=False).head(4)
    transformed = transform_features(raw, load_preprocessor(preprocessor_path))
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(transformed))).squeeze(1).numpy()
    input_path = tmp_path / "features.csv"
    output_path = tmp_path / "predictions.csv"
    pd.DataFrame(transformed, columns=[f"feature_{index}" for index in range(transformed.shape[1])]).to_csv(
        input_path, index=False
    )
    completed = invoke(runner, model_path, input_path, output_path, "cpu")
    assert completed.returncode == 0, completed.stderr
    actual = pd.read_csv(output_path)["fraud_probability"].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
