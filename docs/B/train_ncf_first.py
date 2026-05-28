"""NCF 首次训练脚本 (对应开发文档 2.2 节)

快速验证模式：仅用 5 个 epoch 验证 NCF 能否正常收敛。
正式训练请使用 train_ncf_best.py（完整 50 epoch）。

Usage:
    python docs/B/train_ncf_first.py
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

QUICK_CFG = dict(CONFIG)
QUICK_CFG["model"] = dict(CONFIG["model"])
QUICK_CFG["model"]["epochs"] = 50
QUICK_CFG["model"]["early_stop_patience"] = 5

# 加载全量数据
data_dir = Path("data/processed")
train_df = pd.read_csv(data_dir / "train.csv")
val_df = pd.read_csv(data_dir / "val.csv")
test_df = pd.read_csv(data_dir / "test.csv")
with open(data_dir / "stats.json") as f:
    stats = json.load(f)

n_users, n_items = stats["n_users"], stats["n_items"]
print(f"Dataset: {n_users} users, {n_items} items")
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 创建模型
model = NCF(
    n_users=n_users, n_items=n_items,
    embedding_dim=64, mlp_layers=[64, 32, 16],
).to(device)

# 训练（快速验证模式：5 epoch）
model = train_ncf(model, train_df, val_df, QUICK_CFG, n_items=n_items, device=device)

# 评估
metrics = evaluate_model(model, test_df, train_df, n_items)
print_metrics(metrics, "NCF")

# 保存结果
Path("outputs/models").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "outputs/models/ncf_best.pt")
with open("outputs/ncf_v1.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\nResults saved to outputs/ncf_v1.json")
print("Model saved to outputs/models/ncf_best.pt")