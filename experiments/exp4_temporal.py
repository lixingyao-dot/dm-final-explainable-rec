"""Temporal decay experiment — NCF vs NCF + time-weighted training."""

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
from src.ncf_models.ncf import NCF, train_ncf
from src.temporal import train_ncf_temporal
from src.evaluate import evaluate_model_sampled, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["ncf", "temporal", "all"])
    args = parser.parse_args()

    run_models = set(args.models)
    if "all" in run_models:
        run_models = {"ncf", "temporal"}

    data_dir = Path(CONFIG["data"]["output_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}

    # ── NCF baseline ──
    if "ncf" in run_models:
        print("\n" + "=" * 60)
        print("  NCF (no temporal)")
        print("=" * 60)
        t0 = time.time()
        model = NCF(n_users=n_users, n_items=n_items,
                    embedding_dim=CONFIG["model"]["embedding_dim"],
                    mlp_layers=CONFIG["model"]["ncf_mlp_layers"]).to(device)

        ckpt = Path("outputs/models/ncf_best.pt")
        if not args.train and ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        else:
            model = train_ncf(model, train_df, val_df, CONFIG, n_items=n_items, device=device)

        all_results["NCF"] = evaluate_model_sampled(model, test_df, train_df, n_items)
        print_metrics(all_results["NCF"], "NCF")
        print(f"  Time: {time.time() - t0:.1f}s")

    # ── NCF + temporal ──
    if "temporal" in run_models:
        for decay in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            print(f"\n" + "=" * 60)
            print(f"  NCF + Temporal (decay_lambda={decay})")
            print("=" * 60)
            t0 = time.time()
            model2 = NCF(n_users=n_users, n_items=n_items,
                         embedding_dim=CONFIG["model"]["embedding_dim"],
                         mlp_layers=CONFIG["model"]["ncf_mlp_layers"]).to(device)

            ckpt2 = Path(f"outputs/models/ncf_temporal_l{decay}.pt")
            if not args.train and ckpt2.exists():
                model2.load_state_dict(torch.load(ckpt2, map_location=device, weights_only=True))
            else:
                model2 = train_ncf_temporal(model2, train_df, val_df, CONFIG,
                                            n_items=n_items, device=device, decay_lambda=decay)

            m = evaluate_model_sampled(model2, test_df, train_df, n_items)
            print_metrics(m, f"NCF tempo λ={decay}")
            all_results[f"NCF_temporal_λ={decay}"] = m
            torch.save(model2.state_dict(), ckpt2)
            print(f"  Time: {time.time() - t0:.1f}s")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  TEMPORAL DECAY COMPARISON")
    print("=" * 60)
    first_key = next(iter(all_results.keys()))
    k_vals = sorted(set(int(k.split("@")[1]) for k in all_results[first_key].keys()))
    for metric in ["HitRate", "NDCG"]:
        print(f"\n  {metric}:")
        header = f"  {'Model':<22}" + "".join(f"{'@' + str(k):<12}" for k in k_vals)
        print(header)
        print(f"  {'─' * (22 + 12 * len(k_vals))}")
        for name, res in all_results.items():
            row = f"  {name:<22}" + "".join(f"{res.get(f'{metric}@{k}', 0):<12.4f}" for k in k_vals)
            print(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("outputs") / f"exp_temporal_{timestamp}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
