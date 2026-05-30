"""Experiment 3: Graph Model Comparison — NCF vs LightGCN.

Compares NCF (MLP-based interaction modeling) against LightGCN (graph-based).
RQ3: Can graph structure better model user-item relationships?
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
from src.ncf import NCF, train_ncf
from src.lightgcn import LightGCN, train_lightgcn, build_adj_matrix, _scipy_to_torch
from src.evaluate import evaluate_model, evaluate_model_sampled, print_metrics


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Graph Model Comparison")
    parser.add_argument("--sampled", action="store_true",
                        help="Use sampled evaluation.")
    parser.add_argument("--train", action="store_true",
                        help="Train from scratch.")
    args = parser.parse_args()

    _evaluate = evaluate_model_sampled if args.sampled else evaluate_model
    eval_label = "sampled" if args.sampled else "full"

    data_dir = Path(CONFIG["data"]["output_dir"])

    print("=" * 60)
    print("  Experiment 3: Graph Model Comparison (NCF vs LightGCN)")
    print("  RQ3: Can graph structure better model user-item relationships?")
    print("=" * 60)
    print(f"  Evaluation mode: {eval_label}")

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users = stats["n_users"]
    n_items = stats["n_items"]
    print(f"  Dataset: {n_users} users, {n_items} items")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── 1. NCF ──
    print("\n" + "=" * 60)
    print("  1/2: NCF")
    print("=" * 60)
    t0 = time.time()

    ncf_model = NCF(
        n_users=n_users, n_items=n_items,
        embedding_dim=CONFIG["model"]["embedding_dim"],
        mlp_layers=CONFIG["model"]["ncf_mlp_layers"],
    ).to(device)

    ckpt_path = Path("outputs/models/ncf_best.pt")
    if not args.train and ckpt_path.exists():
        print("  Loading checkpoint ...")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        for name, param in ncf_model.named_parameters():
            if name in state and state[name].shape != param.shape:
                old_shape = state[name].shape
                new = torch.zeros_like(param)
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, param.shape))
                new[slices] = state[name][slices]
                state[name] = new
        ncf_model.load_state_dict(state)
    else:
        ncf_model = train_ncf(ncf_model, train_df, val_df, CONFIG, n_items=n_items, device=device)

    ncf_metrics = _evaluate(ncf_model, test_df, train_df, n_items)
    print_metrics(ncf_metrics, "NCF")
    all_results["NCF"] = ncf_metrics
    torch.save(ncf_model.state_dict(), "outputs/models/ncf_best.pt")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── 2. LightGCN ──
    print("\n" + "=" * 60)
    print("  2/2: LightGCN")
    print("=" * 60)
    t0 = time.time()

    print("  Building normalized adjacency matrix ...")
    adj_norm = build_adj_matrix(train_df, n_users, n_items)
    print(f"  Adj matrix shape: {adj_norm.shape}")

    lightgcn_model = LightGCN(
        n_users=n_users, n_items=n_items,
        embedding_dim=CONFIG["model"]["embedding_dim"],
        n_layers=CONFIG["model"]["lightgcn_layers"],
    ).to(device)

    lightgcn_ckpt = Path("outputs/models/lightgcn_best.pt")
    if not args.train and lightgcn_ckpt.exists():
        print("  Loading checkpoint ...")
        state = torch.load(lightgcn_ckpt, map_location=device, weights_only=True)
        for name, param in lightgcn_model.named_parameters():
            if name in state and state[name].shape != param.shape:
                old_shape = state[name].shape
                new = torch.zeros_like(param)
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, param.shape))
                new[slices] = state[name][slices]
                state[name] = new
        lightgcn_model.load_state_dict(state)
    else:
        lightgcn_model = train_lightgcn(lightgcn_model, train_df, val_df, adj_norm,
                                        CONFIG, device=device)

    # Wrap recommend and score_items for evaluation
    adj_tensor = _scipy_to_torch(adj_norm).to(device)

    original_recommend = lightgcn_model.recommend
    def wrapped_recommend(user_id, n_items, k, exclude=None):
        return original_recommend(user_id, n_items, k, exclude=exclude,
                                  adj_norm=adj_norm, device=device)
    lightgcn_model.recommend = wrapped_recommend

    original_score_items = lightgcn_model.score_items
    def wrapped_score_items(user_id, items):
        return original_score_items(user_id, items, adj_tensor=adj_tensor, device=device)
    lightgcn_model.score_items = wrapped_score_items

    lgcn_metrics = _evaluate(lightgcn_model, test_df, train_df, n_items)
    print_metrics(lgcn_metrics, "LightGCN")
    all_results["LightGCN"] = lgcn_metrics
    torch.save(lightgcn_model.state_dict(), "outputs/models/lightgcn_best.pt")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Summary ──
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 3: GRAPH MODEL COMPARISON SUMMARY")
    print("=" * 60)

    k_values = sorted(set(int(k.split("@")[1]) for k in all_results["NCF"].keys()))
    for metric in ["HitRate", "NDCG"]:
        print(f"\n  {metric}:")
        header = f"  {'Model':<15}" + "".join(f"{'@' + str(k):<12}" for k in k_values)
        print(header)
        print(f"  {'─' * (15 + 12 * len(k_values))}")
        for name, res in all_results.items():
            row = f"  {name:<15}" + "".join(f"{res.get(f'{metric}@{k}', 0):<12.4f}" for k in k_values)
            print(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("outputs") / f"exp3_graph_{eval_label}_{timestamp}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
