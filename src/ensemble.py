"""NCF + ItemCF score-level fusion ensemble."""

import numpy as np
import torch


class NCFItemCFEnsemble:
    """Blend NCF and ItemCF scores: final = (1-alpha)*NCF + alpha*ItemCF.

    Both score vectors are min-max normalized per user before blending.
    """

    def __init__(self, ncf_model, itemcf_model, alpha=0.4, device=None):
        self.ncf = ncf_model
        self.alpha = alpha
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_items = itemcf_model.n_items
        self.n_users = itemcf_model.n_users

        # Precompute ItemCF scores: user_item @ item_sim (sparse-dense matmul, seconds)
        self._itemcf_scores = itemcf_model.user_item @ itemcf_model.similarities

    def _norm(self, scores):
        smin, smax = scores.min(), scores.max()
        if smax > smin:
            return (scores - smin) / (smax - smin)
        return scores

    def score_items(self, user_id, items):
        ncf_raw = self.ncf.score_items(user_id, items, device=self.device)
        ncf_norm = self._norm(ncf_raw)

        # Vectorized: lookup precomputed ItemCF scores, no loop
        itemcf_raw = np.array(self._itemcf_scores[user_id, items].todense()).ravel()
        itemcf_norm = self._norm(itemcf_raw)

        return (1 - self.alpha) * ncf_norm + self.alpha * itemcf_norm

    def recommend(self, user_id, n_items, k, exclude=None):
        """Top-k from blended scores over all items."""
        exclude = exclude or set()
        scores = self.score_items(user_id, list(range(n_items)))
        for item in exclude:
            scores[item] = -999
        top = np.argsort(scores)[::-1][:k]
        return top.tolist()
