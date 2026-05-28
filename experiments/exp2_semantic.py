"""Experiment 2: Semantic Contribution — NCF vs NCF+Review.

Compares NCF (behavioral only) against NCF+Review with different fusion weights.
RQ2: Can review semantics improve recommendation performance?
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.ncf import NCF, train_ncf
from src.ncf_review import NCFReview, train_ncf_review
from src.evaluate import evaluate_model, print_metrics


def main():
    data_dir = Path(CONFIG["data"]["output_dir"])

    print("=" * 60)
    print("  Experiment 2: Semantic Contribution Analysis")
    print("  RQ2: Can review semantics improve recommendation?")
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

    # ── Load SBERT embeddings ──
    print("\n  Loading SBERT embeddings ...")
    user_emb = np.load(data_dir / "user_emb.npy")
    item_emb = np.load(data_dir / "item_emb.npy")
    print(f"  user_emb: {user_emb.shape}, item_emb: {item_emb.shape}")

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── 1. NCF baseline ──
    print("\n" + "=" * 60)
    print("  1/4: Training NCF (baseline, no review semantics)")
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

    # ── 2. NCF+Review alpha=0.1 ──
    print("\n" + "=" * 60)
    print("  2/4: Training NCF+Review (alpha=0.1)")
    print("=" * 60)
    t0 = time.time()

    model_r01 = NCFReview(
        n_users=n_users, n_items=n_items,
        embedding_dim=CONFIG["model"]["embedding_dim"],
        mlp_layers=CONFIG["model"]["ncf_mlp_layers"],
        review_emb_dim=user_emb.shape[1], alpha=0.1,
    ).to(device)

    model_r01 = train_ncf_review(model_r01, train_df, val_df, user_emb, item_emb,
                                  CONFIG, n_items=n_items, device=device)
    model_r01.set_review_embeddings(user_emb, item_emb)

    metrics_r01 = evaluate_model(model_r01, test_df, train_df, n_items)
    print_metrics(metrics_r01, "NCF+Review (alpha=0.1)")
    all_results["NCF+Review_alpha=0.1"] = metrics_r01
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── 3. NCF+Review alpha=0.3 ──
    print("\n" + "=" * 60)
    print("  3/4: Training NCF+Review (alpha=0.3)")
    print("=" * 60)
    t0 = time.time()

    model_r03 = NCFReview(
        n_users=n_users, n_items=n_items,
        embedding_dim=CONFIG["model"]["embedding_dim"],
        mlp_layers=CONFIG["model"]["ncf_mlp_layers"],
        review_emb_dim=user_emb.shape[1], alpha=0.3,
    ).to(device)

    model_r03 = train_ncf_review(model_r03, train_df, val_df, user_emb, item_emb,
                                  CONFIG, n_items=n_items, device=device)
    model_r03.set_review_embeddings(user_emb, item_emb)

    metrics_r03 = evaluate_model(model_r03, test_df, train_df, n_items)
    print_metrics(metrics_r03, "NCF+Review (alpha=0.3)")
    all_results["NCF+Review_alpha=0.3"] = metrics_r03
    torch.save(model_r03.state_dict(), "outputs/models/ncf_review_best.pt")
    print(f"  Best NCF+Review model saved to outputs/models/ncf_review_best.pt")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── 4. NCF+Review alpha=0.5 ──
    print("\n" + "=" * 60)
    print("  4/4: Training NCF+Review (alpha=0.5)")
    print("=" * 60)
    t0 = time.time()

    model_r05 = NCFReview(
        n_users=n_users, n_items=n_items,
        embedding_dim=CONFIG["model"]["embedding_dim"],
        mlp_layers=CONFIG["model"]["ncf_mlp_layers"],
        review_emb_dim=user_emb.shape[1], alpha=0.5,
    ).to(device)

    model_r05 = train_ncf_review(model_r05, train_df, val_df, user_emb, item_emb,
                                  CONFIG, n_items=n_items, device=device)
    model_r05.set_review_embeddings(user_emb, item_emb)

    metrics_r05 = evaluate_model(model_r05, test_df, train_df, n_items)
    print_metrics(metrics_r05, "NCF+Review (alpha=0.5)")
    all_results["NCF+Review_alpha=0.5"] = metrics_r05
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Summary table ──
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 2: SEMANTIC CONTRIBUTION SUMMARY")
    print("=" * 60)

    k_values = sorted(set(int(k.split("@")[1]) for k in all_results["NCF"].keys()))
    metric_names = ["Precision", "Recall", "HitRate", "MAP", "NDCG"]

    for metric in metric_names:
        print(f"\n  {metric}:")
        header = f"  {'Model':<22}" + "".join(f"{'@' + str(k):<12}" for k in k_values)
        print(header)
        print(f"  {'─' * (22 + 12 * len(k_values))}")
        for model_name, res in all_results.items():
            row = f"  {model_name:<22}"
            for k in k_values:
                val = res.get(f"{metric}@{k}", 0)
                row += f"{val:<12.4f}"
            print(row)

    # Save results
    output_path = Path("outputs") / "exp2_semantic_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()