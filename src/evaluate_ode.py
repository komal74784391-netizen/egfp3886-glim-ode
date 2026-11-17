"""Evaluation for GLIM-ODE."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/neurips26.yaml")
    args = parser.parse_args()
    print("Evaluating GLIM-ODE with", args.config)


if __name__ == "__main__":
    main()
