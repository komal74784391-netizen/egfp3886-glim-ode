"""Training for GLIM-ODE."""

import argparse

import torch

from glim_ode.gating import GatedLatentUpdate
from glim_ode.latent import LatentState
from glim_ode.solver import ImplicitStepSolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/neurips26.yaml")
    args = parser.parse_args()
    gating = GatedLatentUpdate()
    latent = LatentState()
    solver = ImplicitStepSolver()
    optimizer = torch.optim.Adam(list(gating.parameters()) + list(latent.parameters()), lr=1e-3)
    print("GLIM-ODE training ready:", args.config, solver.iterations, optimizer.defaults["lr"])


if __name__ == "__main__":
    main()
