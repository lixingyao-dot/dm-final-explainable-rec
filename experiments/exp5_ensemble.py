"""NCF + ItemCF ensemble evaluation."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.ncf_models.ncf import NCF
from src.base_model.itemcf import ItemCF
from src.ensemble import NCFItemCFEnsemble
from src.evaluate import evaluate_model_sampled, print_metrics


def main():
    data_dir = Path(CONFIG["data"]["output_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Dataset: {n_users} users x {n_items} items, Device: {device}")

    all_results = {}

    # ── ItemCF baseline ──
    print("\n  Building ItemCF ...")
    t0 = time.time()
    itemcf = ItemCF(train_df, n_users, n_items)
    m = evaluate_model_sampled(itemcf, test_df, train_df, n_items)
    print_metrics(m, "ItemCF")
    all_results["ItemCF"] = m
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── NCF ──
    print("\n  Loading NCF checkpoint ...")
    ckpt = Path("outputs/models/ncf_best.pt")
    ckpt_state = torch.load(ckpt, map_location=device, weights_only=True)
    ckpt_dim = ckpt_state["user_emb_gmf.weight"].shape[1]
    print(f"  Checkpoint embedding_dim={ckpt_dim}")
    ncf = NCF(
        n_users=n_users, n_items=n_items,
        embedding_dim=ckpt_dim,
        mlp_layers=CONFIG["model"]["ncf_mlp_layers"],
    ).to(device)
    for name, param in ncf.named_parameters():
        if name in ckpt_state and ckpt_state[name].shape != param.shape:
            old_shape = ckpt_state[name].shape
            new = torch.zeros_like(param)
            slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, param.shape))
            new[slices] = ckpt_state[name][slices]
            ckpt_state[name] = new
    ncf.load_state_dict(ckpt_state)
    ncf.eval()

    m = evaluate_model_sampled(ncf, test_df, train_df, n_items)
    print_metrics(m, "NCF")
    all_results["NCF"] = m

    # ── Ensemble alpha sweep (0.1 ~ 0.9) ──
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f"\n  Building NCF+ItemCF ensemble (alpha={alpha}) ...")
        t0 = time.time()
        ens = NCFItemCFEnsemble(ncf, itemcf, alpha=alpha, device=device)
        m = evaluate_model_sampled(ens, test_df, train_df, n_items)
        print_metrics(m, f"NCF+ItemCF (alpha={alpha})")
        all_results[f"NCF+ItemCF_a={alpha}"] = m
        print(f"  Time: {time.time() - t0:.1f}s")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  ENSEMBLE SUMMARY")
    print("=" * 60)
    first = next(iter(all_results.values()))
    k_vals = sorted(set(int(k.split("@")[1]) for k in first.keys()))
    for metric in ["HitRate", "NDCG"]:
        print(f"\n  {metric}:")
        header = f"  {'Model':<22}" + "".join(f"{'@' + str(k):<12}" for k in k_vals)
        print(header)
        print(f"  {'─' * (22 + 12 * len(k_vals))}")
        for name, res in all_results.items():
            row = f"  {name:<22}" + "".join(f"{res.get(f'{metric}@{k}', 0):<12.4f}" for k in k_vals)
            print(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("outputs") / f"exp_ensemble_{timestamp}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
