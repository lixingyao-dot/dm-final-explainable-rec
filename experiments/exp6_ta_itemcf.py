"""Experiment 6: Temporal-Aware ItemCF — decay_lambda sweep.

Evaluates TA-ItemCF across multiple lambda values and compares against
standard ItemCF (lambda=0 equivalent) and NCF+Temporal.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.base_model.itemcf import ItemCF
from src.base_model.ta_itemcf import TAItemCF
from src.evaluate import evaluate_model_sampled, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
    parser.add_argument("--sampled", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = Path(CONFIG["data"]["output_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    print(f"  Dataset: {n_users} users x {n_items} items")
    print(f"  Train interactions: {len(train_df)}")
    print(f"  lambda values: {args.lambdas}")

    all_results = {}

    # ── Standard ItemCF baseline (lambda=0 equivalent) ──
    print("\n" + "=" * 60)
    print("  Standard ItemCF (no temporal weight)")
    print("=" * 60)
    t0 = time.time()
    itemcf = ItemCF(train_df, n_users, n_items)
    m = evaluate_model_sampled(itemcf, test_df, train_df, n_items)
    print_metrics(m, "ItemCF")
    all_results["ItemCF"] = m
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── TA-ItemCF lambda sweep ──
    for decay in args.lambdas:
        print(f"\n" + "=" * 60)
        print(f"  TA-ItemCF (decay_lambda={decay})")
        print("=" * 60)
        t0 = time.time()
        ta_model = TAItemCF(train_df, n_users, n_items, decay_lambda=decay)
        m = evaluate_model_sampled(ta_model, test_df, train_df, n_items)
        print_metrics(m, f"TA-ItemCF λ={decay}")
        all_results[f"TA-ItemCF_λ={decay}"] = m
        print(f"  Time: {time.time() - t0:.1f}s")

    # ── Summary ──
    print("\n\n" + "=" * 60)
    print("  TA-ItemCF SWEEP SUMMARY")
    print("=" * 60)
    first_key = next(iter(all_results.keys()))
    k_vals = sorted(set(int(k.split("@")[1]) for k in all_results[first_key].keys()))
    for metric in ["HitRate", "NDCG"]:
        print(f"\n  {metric}:")
        header = f"  {'Model':<22}" + "".join(f"{'@'+str(k):<12}" for k in k_vals)
        print(header)
        print(f"  {'─'*(22 + 12*len(k_vals))}")
        for name, res in all_results.items():
            row = f"  {name:<22}" + "".join(f"{res.get(f'{metric}@{k}',0):<12.4f}" for k in k_vals)
            print(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("outputs") / f"exp6_ta_itemcf_{timestamp}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
