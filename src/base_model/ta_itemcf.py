"""Temporal-Aware ItemCF: incorporate recency into item similarity computation.

Standard ItemCF treats all interactions equally when computing item-item similarity.
TA-ItemCF weights each interaction by exp(-lambda * (t_max - t) / span), so recent
co-occurrences contribute more to similarity than stale ones.

Intuition: two users who bought the same item at similar times should have higher
collaborative signal than those who bought it years apart.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from .itemcf import ItemCF


class TAItemCF(ItemCF):
    """Temporal-Aware ItemCF: time-weighted item-item similarity."""

    def __init__(self, train_df, n_users, n_items, decay_lambda=2.0, k_neighbors=50):
        self.decay_lambda = decay_lambda
        self.n_users = n_users
        self.n_items = n_items
        self.k_neighbors = k_neighbors

        rows = train_df["user_id"].values
        cols = train_df["item_id"].values

        # ── Compute temporal weights ──
        t = train_df["timestamp"].values.astype(float)
        t_min, t_max = t.min(), t.max()
        span = max(t_max - t_min, 1.0)
        weights = np.exp(-decay_lambda * (t_max - t) / span)

        self.user_item = csr_matrix(
            (weights.astype(np.float32), (rows, cols)),
            shape=(n_users, n_items),
        )
        self.user_item.eliminate_zeros()

        self._build_similarity()
