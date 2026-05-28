"""融合权重 α 消融实验 (对应开发文档 3.4 节)

快速验证模式：每个 α 值仅用 5 个 epoch 快速对比。
正式消融实验请使用 exp2_semantic.py（完整 50 epoch）。

Usage:
    python docs/B/train_ncf_review_alpha.py
"""
import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")
from config import CONFIG
from src.ncf_review import NCFReview, train_ncf_review
from src.evaluate import evaluate_model

QUICK_CFG = dict(CONFIG)
QUICK_CFG["model"] = dict(CONFIG["model"])
QUICK_CFG["model"]["epochs"] = 50
QUICK_CFG["model"]["early_stop_patience"] = 5

data_dir = Path("data/processed")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "val.csv")
test_df = pd.read_csv(data_dir / "test.csv")
with open(data_dir / "stats.json") as f:
    stats = json.load(f)

n_users, n_items = stats["n_users"], stats["n_items"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

user_emb = np.load(data_dir / "user_emb.npy")
item_emb = np.load(data_dir / "item_emb.npy")

alpha_values = [0.0, 0.1, 0.3, 0.5]
all_results = {}

for alpha in alpha_values:
    print(f'\n{"=" * 60}')
    print(f"  Training NCF+Review with alpha={alpha}")
    print(f'{"=" * 60}')

    model = NCFReview(
        n_users=n_users, n_items=n_items,
        embedding_dim=64, mlp_layers=[64, 32, 16],
        review_emb_dim=384, alpha=alpha,
    ).to(device)

    model = train_ncf_review(
        model, train_df, val_df,
        user_emb, item_emb,
        QUICK_CFG, n_items=n_items, device=device,
    )
    model.set_review_embeddings(user_emb, item_emb)

    metrics = evaluate_model(model, test_df, train_df, n_items)
    all_results[f"alpha_{alpha}"] = metrics

# 汇总表
print("\n\n" + "=" * 60)
print("  ALPHA ABLATION SUMMARY")
print("=" * 60)

k_values = sorted(set(int(k.split("@")[1]) for k in all_results["alpha_0.0"].keys()))
metric_names = ["Precision", "Recall", "HitRate", "MAP", "NDCG"]

for metric in metric_names:
    print(f"\n  {metric}:")
    header = f"  {'alpha':<8}" + "".join(f"{'@' + str(k):<12}" for k in k_values)
    print(header)
    print(f"  {'─' * (8 + 12 * len(k_values))}")
    for alpha_key, res in all_results.items():
        row = f"  {alpha_key:<8}"
        for k in k_values:
            val = res.get(f"{metric}@{k}", 0)
            row += f"{val:<12.4f}"
        print(row)

# 保存
Path("outputs/models").mkdir(parents=True, exist_ok=True)
with open("outputs/ncf_review_alpha.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nAlpha ablation results saved to outputs/ncf_review_alpha.json")