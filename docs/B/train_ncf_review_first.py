"""NCF+Review 首次训练脚本 (对应开发文档 3.3 节)

快速验证模式：仅用 5 个 epoch 验证 NCF+Review 能否正常收敛。
正式训练请使用 exp2_semantic.py（完整 50 epoch）。

Usage:
    python docs/B/train_ncf_review_first.py
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
from src.evaluate import evaluate_model, print_metrics

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

# 加载 SBERT embedding
user_emb = np.load(data_dir / "user_emb.npy")
item_emb = np.load(data_dir / "item_emb.npy")
print(f"Loaded SBERT embeddings: user={user_emb.shape}, item={item_emb.shape}")

# 创建模型（α=0.3）
model = NCFReview(
    n_users=n_users, n_items=n_items,
    embedding_dim=64, mlp_layers=[64, 32, 16],
    review_emb_dim=384, alpha=0.3,
).to(device)

# 训练（快速验证模式：5 epoch）
model = train_ncf_review(
    model, train_df, val_df,
    user_emb, item_emb,
    QUICK_CFG, n_items=n_items, device=device,
)

# 存储 embedding 到模型，使 recommend() 可被 evaluate_model 调用
model.set_review_embeddings(user_emb, item_emb)

# 评估
metrics = evaluate_model(model, test_df, train_df, n_items)
print_metrics(metrics, "NCF+Review (alpha=0.3)")

# 保存
Path("outputs/models").mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), "outputs/models/ncf_review_best.pt")
with open("outputs/ncf_review_v1.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Model saved to outputs/models/ncf_review_best.pt")
print("Metrics saved to outputs/ncf_review_v1.json")