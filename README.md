# GLIM-ODE: Gated Latent Implicit Models for Neural ODE Solvers

Official implementation of the GLIM-ODE paper on gated latent implicit
integrators for neural ODE solvers.

## Layout

- `src/glim_ode/gating.py` - gated latent update rule
- `src/glim_ode/latent.py` - latent state parametrization
- `src/glim_ode/solver.py` - implicit ODE step solver
- `src/train_ode.py` - training script
- `src/evaluate_ode.py` - evaluation on ODE benchmarks
- `configs/neurips26.yaml` - paper configuration
- `scripts/run_experiments.sh` - reproducibility script
- `results/README.md` - released results

## Citation

A. Flores, T. Ishikawa, and R. Moreau. "Gated Latent Implicit Models for
Efficient Neural ODE Solvers." NeurIPS 2026.
