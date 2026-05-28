#!/usr/bin/env python3
"""Unified runner for all experiments.

Usage:
    python run_all.py          # Run all 3 experiments sequentially
    python run_all.py --exp 1  # Run only experiment 1
    python run_all.py --exp 2  # Run only experiment 2
    python run_all.py --exp 3  # Run only experiment 3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(step_name: str, command: list):
    print(f"\n{'#' * 70}")
    print(f"#  STEP: {step_name}")
    print(f"#  CMD:  {' '.join(command)}")
    print(f"{'#' * 70}\n")
    t0 = time.time()
    result = subprocess.run(command, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ STEP FAILED: {step_name} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  ✓ Step completed in {elapsed:.1f}s")
    return result


def setup_output_dirs():
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/ncf_hyper").mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run recommendation experiments")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], default=None,
                        help="Run a specific experiment (1, 2, or 3). Default: run all.")
    args = parser.parse_args()

    setup_output_dirs()

    print("=" * 70)
    print("  RECOMMENDATION SYSTEM — EXPERIMENT PIPELINE")
    print("  Project: Explainable Recommendation with User Behavior & Reviews")
    print("=" * 70)

    experiments = {
        1: ("Experiment 1: Baseline Comparison (Popularity / UserCF / ItemCF / NCF)",
            [sys.executable, "experiments/exp1_baseline.py"]),
        2: ("Experiment 2: Semantic Contribution (NCF vs NCF+Review)",
            [sys.executable, "experiments/exp2_semantic.py"]),
        3: ("Experiment 3: Graph Model Comparison (NCF vs LightGCN)",
            [sys.executable, "experiments/exp3_graph.py"]),
    }

    if args.exp is not None:
        name, cmd = experiments[args.exp]
        print(f"\n  Running: {name}")
        run_step(name, cmd)
    else:
        for exp_id in [1, 2, 3]:
            name, cmd = experiments[exp_id]
            run_step(name, cmd)

    print(f"\n{'=' * 70}")
    print("  ALL REQUESTED EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"{'=' * 70}")
    print("\n  Output files:")
    print("    outputs/exp1_baseline_results.json")
    print("    outputs/exp2_semantic_results.json")
    print("    outputs/exp3_graph_results.json")
    print("    outputs/models/ncf_best.pt")
    print("    outputs/models/ncf_review_best.pt")
    print("    outputs/models/lightgcn_best.pt")


if __name__ == "__main__":
    main()