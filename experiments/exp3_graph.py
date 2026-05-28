"""Experiment 3: Graph Model Comparison — NCF vs LightGCN.

Compares NCF (MLP-based interaction modeling) against LightGCN (graph-based).
RQ3: Can graph structure better model user-item relationships?
"""

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.ncf import NCF, train_ncf
from src.lightgcn import LightGCN, train_lightgcn, build_adj_matrix
from src.evaluate import evaluate_model, print_metrics


def main():
    data_dir = Path(CONFIG["data"]["output_dir"])

    print("=" * 60)
    print("  Experiment 3: Graph Model Comparison (NCF vs LightGCN)")
    print("  RQ3: Can graph structure better model user-item relationships?")
    print("=" * 60)

    # ── Load data ──
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
    print("  1/2: Training NCF")
    print("=" * 60)
    t0 = time.time()

    ncf_model = NCF(
        n_users=n_users, n_items=n_items,
        embedding_dim=CONFIG["model"]["embedding_dim"],
        mlp_layers=CONFIG["model"]["ncf_mlp_layers"],
    ).to(device)

    ncf_model = train_ncf(ncf_model, train_df, val_df, CONFIG, n_items=n_items, device=device)

    ncf_metrics = evaluate_model(ncf_model, test_df, train_df, n_items)
    print_metrics(ncf_metrics, "NCF")
    all_results["NCF"] = ncf_metrics

    torch.save(ncf_model.state_dict(), "outputs/models/ncf_best.pt")
    print(f"  NCF model saved to outputs/models/ncf_best.pt")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── 2. LightGCN ──
    print("\n" + "=" * 60)
    print("  2/2: Training LightGCN")
    print("=" * 60)
    t0 = time.time()

    # LightGCN 需要更多 epoch 收敛，使用独立配置
    lightgcn_cfg = deepcopy(CONFIG)
    lightgcn_cfg["model"]["epochs"] = 200

    print("  Building normalized adjacency matrix ...")
    adj_norm = build_adj_matrix(train_df, n_users, n_items)
    print(f"  Adj matrix shape: {adj_norm.shape}")

    lightgcn_model = LightGCN(
        n_users=n_users, n_items=n_items,
        embedding_dim=CONFIG["model"]["embedding_dim"],
        n_layers=CONFIG["model"]["lightgcn_layers"],
    ).to(device)

    lightgcn_model = train_lightgcn(
        lightgcn_model, train_df, val_df,
        adj_norm, lightgcn_cfg, device=device
    )

    # Wrap recommend to include adj_norm and device for evaluate_model compatibility
    original_recommend = lightgcn_model.recommend

    def wrapped_recommend(user_id, n_items, k, exclude=None):
        return original_recommend(
            user_id, n_items, k,
            exclude=exclude,
            adj_norm=adj_norm,
            device=device,
        )

    lightgcn_model.recommend = wrapped_recommend

    lgcn_metrics = evaluate_model(lightgcn_model, test_df, train_df, n_items)
    print_metrics(lgcn_metrics, "LightGCN")
    all_results["LightGCN"] = lgcn_metrics

    torch.save(lightgcn_model.state_dict(), "outputs/models/lightgcn_best.pt")
    print(f"  LightGCN model saved to outputs/models/lightgcn_best.pt")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Summary table ──
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 3: GRAPH MODEL COMPARISON SUMMARY")
    print("=" * 60)

    k_values = sorted(set(int(k.split("@")[1]) for k in all_results["NCF"].keys()))
    metric_names = ["Precision", "Recall", "HitRate", "MAP", "NDCG"]

    for metric in metric_names:
        print(f"\n  {metric}:")
        header = f"  {'Model':<15}" + "".join(f"{'@' + str(k):<12}" for k in k_values)
        print(header)
        print(f"  {'─' * (15 + 12 * len(k_values))}")
        for model_name, res in all_results.items():
            row = f"  {model_name:<15}"
            for k in k_values:
                val = res.get(f"{metric}@{k}", 0)
                row += f"{val:<12.4f}"
            print(row)

    output_path = Path("outputs") / "exp3_graph_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()