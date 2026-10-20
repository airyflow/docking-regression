"""Surrogate model: embedding -> docking score.

Deliberately plain. The question this repo asks is which molecular backbone
carries the signal and what fusing them buys, so the head is held fixed and
cheap across every backbone from 512 to 3400 dims.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(1024, 512), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(in_dim: int, hidden=(1024, 512), dropout: float = 0.1) -> MLP:
    return MLP(in_dim, hidden=hidden, dropout=dropout)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
