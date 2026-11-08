"""Fraud-model definition and export to the portable C++ MIE_MLP_V1 format."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


MODEL_DIMENSIONS = (30, 64, 32, 1)


class FraudMLP(nn.Module):
    """The PyTorch architecture mirrored exactly by the C++ dense backends."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(MODEL_DIMENSIONS[0], MODEL_DIMENSIONS[1]),
                nn.Linear(MODEL_DIMENSIONS[1], MODEL_DIMENSIONS[2]),
                nn.Linear(MODEL_DIMENSIONS[2], MODEL_DIMENSIONS[3]),
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.layers[0](inputs))
        hidden = torch.relu(self.layers[1](hidden))
        return self.layers[2](hidden)


def export_engine_model(model: FraudMLP, path: str | Path) -> None:
    """Write FP32 row-major parameters understood by ``mie::load_mlp_file``.

    ``nn.Linear.weight`` is already [out_features, in_features], which is the
    exact row-major layout used by the C++ reference, OpenMP, and CUDA paths.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["MIE_MLP_V1", "activation sigmoid", f"layers {len(model.layers)}"]
    with torch.no_grad():
        for layer in model.layers:
            weight = layer.weight.detach().cpu().to(torch.float32).contiguous().view(-1).tolist()
            bias = layer.bias.detach().cpu().to(torch.float32).contiguous().view(-1).tolist()
            lines.append(f"layer {layer.in_features} {layer.out_features}")
            lines.append(f"weights {len(weight)}")
            lines.extend(format(float(value), ".9g") for value in weight)
            lines.append(f"biases {len(bias)}")
            lines.extend(format(float(value), ".9g") for value in bias)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
