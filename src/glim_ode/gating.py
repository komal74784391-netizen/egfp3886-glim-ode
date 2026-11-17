"""Gated latent update."""

import torch
from torch import nn


class GatedLatentUpdate(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.candidate = nn.Linear(dim * 2, dim)

    def forward(self, state: torch.Tensor, drift: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([state, drift], dim=-1)))
        candidate = torch.tanh(self.candidate(torch.cat([state, drift], dim=-1)))
        return gate * state + (1 - gate) * candidate
