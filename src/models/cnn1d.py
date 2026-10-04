"""Compact 1D CNN for beat-centered, two-channel ECG segments."""
import torch
from torch import nn


class ECG1DCNN(nn.Module):
    """Three Conv1D blocks with global pooling and a 15-class-ready head.

    Input tensors use channels-first shape ``[batch, channels, samples]``.
    """

    def __init__(self, in_channels=2, num_classes=15, input_length=216, dropout=0.3):
        super().__init__()
        if in_channels <= 0 or num_classes < 2 or input_length < 4:
            raise ValueError("Invalid model dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.input_length = int(input_length)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(64, num_classes))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[2] != self.input_length:
            raise ValueError(
                f"Expected [batch, {self.in_channels}, {self.input_length}], got {tuple(x.shape)}"
            )
        return self.classifier(self.features(x).squeeze(-1))


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_architecture():
    return {
        "input_tensor_shape": ["batch", 2, 216],
        "layers": [
            {"layer": "Conv1d", "in_channels": 2, "out_channels": 16, "kernel_size": 7, "padding": 3, "bias": False},
            {"layer": "BatchNorm1d", "features": 16},
            {"layer": "ReLU"},
            {"layer": "MaxPool1d", "kernel_size": 2},
            {"layer": "Conv1d", "in_channels": 16, "out_channels": 32, "kernel_size": 5, "padding": 2, "bias": False},
            {"layer": "BatchNorm1d", "features": 32},
            {"layer": "ReLU"},
            {"layer": "MaxPool1d", "kernel_size": 2},
            {"layer": "Conv1d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "padding": 1, "bias": False},
            {"layer": "BatchNorm1d", "features": 64},
            {"layer": "ReLU"},
            {"layer": "AdaptiveAvgPool1d", "output_size": 1},
            {"layer": "Dropout", "p": 0.3},
            {"layer": "Linear", "in_features": 64, "out_features": 15},
        ],
        "parameter_count_depends_on_num_classes": True,
    }
