"""Baseline recommendation models."""

import numpy as np
from collections import Counter


class PopularityRecommender:
    """Recommend the globally most popular items to all users."""

    def __init__(self, train_df):
        item_counts = Counter(train_df["item_id"])
        self.popular_items = [item for item, _ in item_counts.most_common()]

    def recommend(self, user_id, n_items, k, exclude=None):
        """Return top-k most popular items, skipping already-interacted items."""
        exclude = exclude or set()
        recs = []
        for item in self.popular_items:
            if item not in exclude:
                recs.append(item)
                if len(recs) >= k:
                    return recs

        # fallback: fill remaining slots with random unexcluded items
        seen = exclude | set(recs)
        remaining = [i for i in range(n_items) if i not in seen]
        np.random.shuffle(remaining)
        recs.extend(remaining[:k - len(recs)])
        return recs
