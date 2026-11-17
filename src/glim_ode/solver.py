"""Implicit ODE step solver."""

import torch


class ImplicitStepSolver:
    def __init__(self, iterations: int = 3, damping: float = 0.5) -> None:
        self.iterations = iterations
        self.damping = damping

    def step(self, state: torch.Tensor, drift_fn) -> torch.Tensor:
        next_state = state
        for _ in range(self.iterations):
            next_state = state + self.damping * drift_fn(next_state)
        return next_state
