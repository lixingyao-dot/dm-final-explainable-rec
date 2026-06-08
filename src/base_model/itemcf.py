"""Item-based Collaborative Filtering."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemCF:
    """Item-based CF: recommend items similar to those the user already interacted with."""

    def __init__(self, train_df, n_users, n_items, k_neighbors=50):
        self.n_users = n_users
        self.n_items = n_items
        self.k_neighbors = k_neighbors

        rows = train_df["user_id"].values
        cols = train_df["item_id"].values
        data = np.ones(len(train_df), dtype=np.float32)
        self.user_item = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        self.user_item.eliminate_zeros()

        self._build_similarity()

    def _build_similarity(self):
        """Compute cosine similarity between items (column-wise)."""
        item_user = self.user_item.T.tocsr()  # (n_items, n_users)
        self.similarities = cosine_similarity(item_user, dense_output=False).tocsr()

    def recommend(self, user_id, n_items, k, exclude=None):
        exclude = exclude or set()

        if user_id >= self.n_users:
            return list(range(k))

        # Items this user has interacted with
        user_row = self.user_item[user_id].toarray().ravel()
        interacted = np.where(user_row > 0)[0]

        if len(interacted) == 0:
            popularity = np.array(self.user_item.sum(axis=0)).ravel()
            candidates = np.argsort(popularity)[::-1]
            recs = []
            for item in candidates:
                if item not in exclude:
                    recs.append(int(item))
                    if len(recs) >= k:
                        return recs
            return recs

        # Score: for each item, sum similarities to items user interacted with
        scores = np.zeros(n_items, dtype=np.float64)
        for j in interacted:
            sim_col = self.similarities[j].toarray().ravel()
            sim_col[j] = 0  # self
            scores += sim_col

        # Exclude
        for item in exclude:
            scores[item] = -1

        top_items = np.argsort(scores)[::-1][:k]
        return [int(i) for i in top_items]

    def explain(self, user_id, item_id, k=5):
        """Explain a recommendation using similar historical items."""
        if user_id >= self.n_users or item_id >= self.n_items:
            return {
                "recommended_item": int(item_id),
                "bridge_items": [],
                "explanation": "Not enough user or item history to explain this recommendation.",
            }

        user_row = self.user_item[user_id].toarray().ravel()
        interacted = np.where(user_row > 0)[0]
        if len(interacted) == 0:
            return {
                "recommended_item": int(item_id),
                "bridge_items": [],
                "explanation": "No user history found for explanation.",
            }

        item_sims = self.similarities[item_id].toarray().ravel()
        bridge_items = []
        for hist_item in interacted:
            if hist_item == item_id:
                continue
            bridge_items.append(
                {
                    "item_id": int(hist_item),
                    "similarity": float(item_sims[hist_item]) if hist_item < len(item_sims) else 0.0,
                }
            )

        bridge_items.sort(key=lambda x: x["similarity"], reverse=True)
        bridge_items = bridge_items[:k]

        if bridge_items:
            bridge_desc = "、".join(
                f"{x['item_id']}(相似度={x['similarity']:.2f})" for x in bridge_items
            )
            explanation = (
                f"推荐该商品是因为你购买过 {bridge_desc}，"
                f"而购买这些商品的用户通常也购买 {item_id}。"
            )
        else:
            explanation = f"推荐该商品是因为它与你的历史购买记录相似。"

        return {
            "recommended_item": int(item_id),
            "bridge_items": bridge_items,
            "explanation": explanation,
        }
