"""User-based Collaborative Filtering."""

import numpy as np
from scipy.sparse import csr_matrix


def _cosine_similarity_sparse(mat):
    dot = mat @ mat.T
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
    denom = np.outer(norms, norms)
    denom[denom == 0] = 1.0
    return dot.multiply(1.0 / denom)


class UserCF:
    """User-based CF: find similar users and recommend items they liked."""

    def __init__(self, train_df, n_users, n_items, k_neighbors=50):
        self.n_users = n_users
        self.n_items = n_items
        self.k_neighbors = k_neighbors

        # Build sparse user-item interaction matrix (implicit feedback: 1 if interacted)
        rows = train_df["user_id"].values
        cols = train_df["item_id"].values
        data = np.ones(len(train_df), dtype=np.float32)
        self.user_item = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        self.user_item.eliminate_zeros()

        # Precompute user-user similarity
        self._build_similarity()

    def _build_similarity(self):
        """Compute cosine similarity between all users."""
        self.similarities = _cosine_similarity_sparse(self.user_item).tocsr()  # sparse

    def recommend(self, user_id, n_items, k, exclude=None):
        """Recommend k items based on similar users' interactions."""
        exclude = exclude or set()

        if user_id >= self.n_users:
            return list(range(k))

        # Get similarities to all other users
        sim_row = self.similarities[user_id].toarray().ravel()
        sim_row[user_id] = 0  # self

        # Find top-k similar users
        neighbor_indices = np.argsort(sim_row)[::-1][:self.k_neighbors]
        neighbor_weights = sim_row[neighbor_indices]

        # Only keep neighbors with positive similarity
        mask = neighbor_weights > 0
        neighbor_indices = neighbor_indices[mask]
        neighbor_weights = neighbor_weights[mask]

        if len(neighbor_indices) == 0:
            # cold start: return popular items
            popularity = np.array(self.user_item.sum(axis=0)).ravel()
            candidates = np.argsort(popularity)[::-1]
            recs = []
            for item in candidates:
                if item not in exclude:
                    recs.append(int(item))
                    if len(recs) >= k:
                        return recs
            return recs

        # Score items: weighted sum of neighbor interactions
        neighbor_matrix = self.user_item[neighbor_indices].toarray()  # (k_neighbors, n_items)
        scores = neighbor_weights.dot(neighbor_matrix)  # (n_items,)

        # Exclude items user already interacted with
        for item in exclude:
            scores[item] = -1

        top_items = np.argsort(scores)[::-1][:k]
        return [int(i) for i in top_items]
