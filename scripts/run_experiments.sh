#!/usr/bin/env bash
set -euo pipefail
for seed in 11 22 33; do
  python src/train_ode.py --config configs/neurips26.yaml
done
