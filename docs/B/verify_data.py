"""数据完整性验证脚本 (对应开发文档 1.4 节)

验证 data/processed/ 目录下是否存在所有必需的数据文件。

Usage:
    python docs/B/verify_data.py
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

d = Path("data/processed")

print("=" * 60)
print("  数据完整性验证")
print("=" * 60)

# 检查 CSV 文件
for fname in ["train.csv", "val.csv", "test.csv"]:
    fpath = d / fname
    if fpath.exists():
        df = pd.read_csv(fpath)
        print(f"  {fname}: {df.shape}")
    else:
        print(f"  {fname}: MISSING!")

# 检查 stats.json
stats_path = d / "stats.json"
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)
    print(f"  stats: {stats}")
else:
    print("  stats.json: MISSING!")

# 检查 SBERT embedding
for fname in ["user_emb.npy", "item_emb.npy"]:
    fpath = d / fname
    if fpath.exists():
        arr = np.load(fpath)
        nan_count = np.any(np.isnan(arr))
        zero_rows = np.all(arr == 0, axis=1).sum()
        print(f"  {fname}: {arr.shape}, dtype={arr.dtype}, "
              f"NaN={nan_count}, zero_rows={zero_rows}")
    else:
        print(f"  {fname}: MISSING!")

# 检查其他辅助文件
for fname in ["user_reviews.json", "item_reviews.json",
              "interactions.csv", "user_map.json", "item_map.json"]:
    fpath = d / fname
    if fpath.exists():
        print(f"  {fname}: OK")
    else:
        print(f"  {fname}: MISSING (optional)")

print("=" * 60)
print("  验证完成")
print("=" * 60)