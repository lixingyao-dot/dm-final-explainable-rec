"""NCF 超参数网格搜索 (对应开发文档 2.3 节)

快速验证模式：每组超参仅用 5 个 epoch 快速筛选最优参数组合。
正式调参请手动修改 QUICK_EPOCHS 为完整值。

Usage:
    python docs/B/train_ncf_hyper.py
"""
import sys
import json
import torch
import pandas as pd
import itertools
import time
from pathlib import Path

sys.path.insert(0, ".")
from config import CONFIG
from src.ncf import NCF, train_ncf
from src.evaluate import evaluate_model

QUICK_EPOCHS = 50
QUICK_PATIENCE = 5

data_dir = Path("data/processed")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "val.csv")
test_df = pd.read_csv(data_dir / "test.csv")
with open(data_dir / "stats.json") as f:
    stats = json.load(f)

n_users, n_items = stats["n_users"], stats["n_items"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

search_space = {
    "embedding_dim": [32, 64, 128],
    "lr": [0.001, 0.0005],
    "mlp_layers": [[64, 32, 16], [128, 64, 32]],
}

keys = list(search_space.keys())
results = []
output_dir = Path("outputs/ncf_hyper")
output_dir.mkdir(parents=True, exist_ok=True)

for values in itertools.product(*[search_space[k] for k in keys]):
    params = dict(zip(keys, values))
    tag = (f"d{params['embedding_dim']}_lr{params['lr']}_"
           f"mlp{'x'.join(map(str, params['mlp_layers']))}")
    print(f"\n=== NCF Hyper: {tag} ===")
    t0 = time.time()

    cfg = dict(CONFIG)
    cfg["model"] = dict(CONFIG["model"])
    cfg["model"]["embedding_dim"] = params["embedding_dim"]
    cfg["model"]["learning_rate"] = params["lr"]
    cfg["model"]["ncf_mlp_layers"] = params["mlp_layers"]
    cfg["model"]["epochs"] = QUICK_EPOCHS
    cfg["model"]["early_stop_patience"] = QUICK_PATIENCE

    model = NCF(
        n_users=n_users, n_items=n_items,
        embedding_dim=params["embedding_dim"],
        mlp_layers=params["mlp_layers"],
    ).to(device)

    model = train_ncf(model, train_df, val_df, cfg, n_items=n_items, device=device)
    metrics = evaluate_model(model, test_df, train_df, n_items)

    entry = {
        "params": params,
        "metrics": metrics,
        "time": round(time.time() - t0, 1),
    }
    results.append(entry)

    with open(output_dir / f"{tag}.json", "w") as f:
        json.dump(entry, f, indent=2)
    print(f"  Saved to {output_dir / tag}.json")

# 汇总
summary = sorted(results, key=lambda x: -x["metrics"].get("NDCG@10", 0))
print("\n\n=== NCF Hyperparameter Summary (sorted by NDCG@10) ===")
header = f"{'Rank':<6}{'embedding_dim':<16}{'lr':<10}{'mlp_layers':<20}{'NDCG@10':<12}{'Recall@10':<12}"
print(header)
print("-" * 76)
for i, r in enumerate(summary):
    p = r["params"]
    m = r["metrics"]
    print(f"{i+1:<6}{p['embedding_dim']:<16}{p['lr']:<10}"
          f"{str(p['mlp_layers']):<20}{m.get('NDCG@10', 0):<12.4f}{m.get('Recall@10', 0):<12.4f}")

with open(output_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nHyperparameter summary saved to {output_dir / 'summary.json'}")