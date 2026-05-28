"""NCF 最优模型训练 (对应开发文档 2.4 节)

使用超参数调优选出的最优参数训练 NCF 模型。
最优参数：embedding_dim=128, lr=0.001, mlp_layers=[128, 64, 32]

Usage:
    python docs/B/train_ncf_best.py
"""
import sys
import json
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
from config import CONFIG
from src.ncf import NCF, train_ncf
from src.evaluate import evaluate_model, print_metrics

data_dir = Path("data/processed")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "val.csv")
test_df = pd.read_csv(data_dir / "test.csv")
with open(data_dir / "stats.json") as f:
    stats = json.load(f)

n_users, n_items = stats["n_users"], stats["n_items"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 超参数调优结果：embedding_dim=128, lr=0.001, mlp_layers=[128, 64, 32]
BEST_CFG = dict(CONFIG)
BEST_CFG["model"] = dict(CONFIG["model"])
BEST_CFG["model"]["learning_rate"] = 0.001
BEST_CFG["model"]["ncf_mlp_layers"] = [128, 64, 32]

model = NCF(
    n_users=n_users, n_items=n_items,
    embedding_dim=128, mlp_layers=[128, 64, 32],
).to(device)

model = train_ncf(model, train_df, val_df, BEST_CFG, n_items=n_items, device=device)
torch.save(model.state_dict(), "outputs/models/ncf_best.pt")

metrics = evaluate_model(model, test_df, train_df, n_items)
print_metrics(metrics, "NCF Best")

with open("outputs/ncf_best_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Model saved to outputs/models/ncf_best.pt")
print("Metrics saved to outputs/ncf_best_metrics.json")