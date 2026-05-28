"""SBERT Embedding 验证脚本 (对应开发文档 3.2 节)

验证 user_emb.npy 和 item_emb.npy 的完整性。

Usage:
    python docs/B/verify_sbert.py
"""
import numpy as np
from pathlib import Path

data_dir = Path("data/processed")

print("=" * 60)
print("  SBERT Embedding 验证")
print("=" * 60)

for fname in ["user_emb.npy", "item_emb.npy"]:
    fpath = data_dir / fname
    if not fpath.exists():
        print(f"  {fname}: FILE NOT FOUND")
        continue

    arr = np.load(fpath)
    print(f"\n  {fname}:")
    print(f"    Shape: {arr.shape}")
    print(f"    Dtype: {arr.dtype}")
    print(f"    Range: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"    Mean:  {arr.mean():.6f}")
    print(f"    Std:   {arr.std():.6f}")
    print(f"    NaN:   {np.any(np.isnan(arr))}")
    print(f"    Inf:   {np.any(np.isinf(arr))}")
    print(f"    Zero rows: {np.all(arr == 0, axis=1).sum()}")

print("\n" + "=" * 60)
print("  验证完成")
print("=" * 60)