"""Latent state parametrization."""

import torch
from torch import nn


class LatentState(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.to_latent = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.to_latent(hidden))
