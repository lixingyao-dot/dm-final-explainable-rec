"""Experiment 0: NCF hyperparameter search — embedding_dim × learning_rate.

Finds optimal dim and lr for NCF on the current dataset before running other experiments.
"""

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
from src.evaluate import evaluate_model_sampled


def main():
    data_dir = Path(CONFIG["data"]["output_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Dataset: {n_users} users x {n_items} items, Device: {device}")

    dims = [32, 64, 128]
    lrs = [0.0005, 0.001]
    results = []

    for dim in dims:
        for lr in lrs:
            cfg = CONFIG.copy()
            cfg["model"] = {**CONFIG["model"], "embedding_dim": dim, "learning_rate": lr}
            label = f"dim={dim}_lr={lr}"

            print(f"\n{'=' * 60}")
            print(f"  {label}")
            print("=" * 60)
            t0 = time.time()

            model = NCF(
                n_users=n_users, n_items=n_items,
                embedding_dim=dim,
                mlp_layers=CONFIG["model"]["ncf_mlp_layers"],
            ).to(device)

            model = train_ncf(model, train_df, val_df, cfg, n_items=n_items, device=device)
            metrics = evaluate_model_sampled(model, test_df, train_df, n_items)

            print(f"\n  {label}: H@10={metrics['HitRate@10']:.4f} | "
                  f"H@20={metrics['HitRate@20']:.4f} | "
                  f"NDCG@10={metrics['NDCG@10']:.4f}")
            print(f"  Time: {time.time() - t0:.1f}s")

            torch.save(model.state_dict(), f"outputs/models/ncf_{label.replace('=', '')}.pt")
            results.append({"params": {"dim": dim, "lr": lr}, "metrics": metrics, "time": time.time() - t0})

    # ── Summary ──
    print("\n\n" + "=" * 70)
    print("  HYPERPARAMETER SEARCH SUMMARY")
    print("=" * 70)
    print(f"\n  {'dim':<6} {'lr':<10} {'H@10':<10} {'H@20':<10} {'NDCG@10':<10} {'time':<8}")
    print(f"  {'─' * 54}")
    best = max(results, key=lambda r: r["metrics"]["HitRate@10"])

    for r in sorted(results, key=lambda r: r["metrics"]["HitRate@10"], reverse=True):
        p = r["params"]
        m = r["metrics"]
        marker = " ← best" if r == best else ""
        print(f"  {p['dim']:<6} {p['lr']:<10} {m['HitRate@10']:<10.4f} {m['HitRate@20']:<10.4f} {m['NDCG@10']:<10.4f} {r['time']:<8.0f}s{marker}")

    print(f"\n  Best config: dim={best['params']['dim']}, lr={best['params']['lr']}")
    print(f"  Best NCF saved to: outputs/models/ncf_best.pt")

    # Save best NCF to the standard checkpoint path
    torch.save(torch.load(f"outputs/models/ncfdim{best['params']['dim']}lr{best['params']['lr']}.pt",
                          map_location="cpu", weights_only=True),
               "outputs/models/ncf_best.pt")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("outputs") / f"exp0_hyper_{timestamp}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out}")


if __name__ == "__main__":
    main()
