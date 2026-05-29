"""Improved configuration — overrides for better NCF performance.

- embedding_dim: 64 → 32  (fewer params on small data, less overfitting)
- neg_ratio: 4 → 9       (train-test distribution alignment)
"""

from config import CONFIG as _BASE

CONFIG = _BASE.copy()
CONFIG["model"] = {**_BASE["model"], "embedding_dim": 32}
CONFIG["negative_sampling"] = {**_BASE["negative_sampling"], "neg_ratio": 9}
