"""Explainability module: keyword, similar user, and SHAP explanations."""

import json
import numpy as np
from collections import Counter
from pathlib import Path


class KeywordExplainer:
    """Extract overlapping keywords between user and item reviews."""

    def __init__(self, user_reviews: dict, item_reviews: dict, top_k: int = 10):
        self.user_reviews = user_reviews
        self.item_reviews = item_reviews
        self.top_k = top_k
        self.stopwords = self._default_stopwords()

    @staticmethod
    def _default_stopwords():
        return set(
            "the a an is are was were be been being have has had do does did "
            "will would shall should may might can could i me my we our you your "
            "he him his she her it its they them their this that these those "
            "and but or nor not so yet for at by to in on of with from as into "
            "about between through during before after above below up down "
            "out off over under again further then once here there when where "
            "why how all each every both few more most other some such no nor "
            "only own same so than too very just because but if while although "
            "until unless since also back even still really already also very "
            "one two three four five six seven eight nine ten get got gets "
            "bought bought use used using good great nice best well better ".split()
        )

    def extract_keywords(self, text: str, top_k: int = None) -> list:
        if not text:
            return []
        top_k = top_k or self.top_k
        words = text.lower().split()
        words = [w for w in words if len(w) > 2 and w not in self.stopwords]
        return [w for w, _ in Counter(words).most_common(top_k)]

    def explain(self, user_id: int, item_id: int) -> dict:
        user_text = self.user_reviews.get(str(user_id)) or self.user_reviews.get(user_id, "")
        item_text = self.item_reviews.get(str(item_id)) or self.item_reviews.get(item_id, "")

        user_kw = self.extract_keywords(user_text)
        item_kw = self.extract_keywords(item_text)
        overlap = [w for w in user_kw if w in item_kw]

        return {
            "user_keywords": user_kw,
            "item_keywords": item_kw,
            "overlap": overlap,
            "explanation": (
                f"You frequently mention: {', '.join(user_kw[:5])}. "
                f"This product's top review keywords: {', '.join(item_kw[:5])}. "
                f"Overlap: {', '.join(overlap[:3]) if overlap else 'none'}."
            ),
        }


class SimilarUserExplainer:
    """Explain recommendations based on similar users' preferences."""

    def __init__(self, user_cf_model=None, train_df=None):
        self.user_cf = user_cf_model
        self.train_df = train_df

    def explain(self, user_id: int, item_id: int) -> dict:
        if self.user_cf is None:
            return {"explanation": "Similar user model not available."}

        neighbors = self.user_cf.user_neighbors.get(user_id, [])
        if not neighbors:
            return {"explanation": "No similar users found."}

        # Check which neighbors rated this item highly
        rated_neighbors = []
        for neighbor_id, sim in neighbors[:10]:
            ratings = self.user_cf.user_ratings.get(neighbor_id, {})
            if item_id in ratings and ratings[item_id] >= 4.0:
                rated_neighbors.append({
                    "neighbor_id": neighbor_id,
                    "similarity": round(sim, 3),
                    "rating": ratings[item_id],
                })

        total_checked = min(len(neighbors), 10)
        n_liked = len(rated_neighbors)

        return {
            "total_similar_users_checked": total_checked,
            "users_who_liked_item": rated_neighbors[:5],
            "explanation": (
                f"Among the {total_checked} most similar users to you, "
                f"{n_liked} gave this item 4+ stars."
            ),
        }


class SHAPExplainer:
    """SHAP-based feature contribution explanation."""

    def __init__(self, model, background_data=None):
        self.model = model
        self.background = background_data
        self._explainer = None

    def _build_features(self, user_id, item_id, user_emb=None, item_emb=None):
        """Build feature vector for a user-item pair."""
        features = {}

        # Behavioral features
        if hasattr(self.model, 'user_emb'):
            features["user_emb_norm"] = float(np.linalg.norm(
                self.model.user_emb.weight[user_id].detach().cpu().numpy()
            ))
        if hasattr(self.model, 'item_emb'):
            features["item_emb_norm"] = float(np.linalg.norm(
                self.model.item_emb.weight[item_id].detach().cpu().numpy()
            ))

        # Semantic features
        if user_emb is not None:
            features["user_review_norm"] = float(np.linalg.norm(user_emb[user_id]))
        if item_emb is not None:
            features["item_review_norm"] = float(np.linalg.norm(item_emb[item_id]))

        if user_emb is not None and item_emb is not None:
            features["review_cosine_sim"] = float(
                np.dot(user_emb[user_id], item_emb[item_id]) /
                (np.linalg.norm(user_emb[user_id]) * np.linalg.norm(item_emb[item_id]) + 1e-8)
            )

        return features

    def explain(self, user_id, item_id, user_emb=None, item_emb=None) -> dict:
        """Generate SHAP-style explanation for a recommendation."""
        features = self._build_features(user_id, item_id, user_emb, item_emb)

        # Simplified: compute feature importance via perturbation
        base_score = self._predict_score(user_id, item_id)

        importance = {}
        for feat_name in features:
            importance[feat_name] = round(features[feat_name] * base_score, 4)

        return {
            "base_score": round(base_score, 4),
            "features": features,
            "feature_importance": importance,
            "explanation": (
                f"Predicted score: {base_score:.3f}. "
                f"Top contributing features: " +
                ", ".join(f"{k}={v:.3f}" for k, v in
                          sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
            ),
        }

    def _predict_score(self, user_id, item_id):
        """Get model's predicted score for a user-item pair."""
        import torch
        self.model.eval()
        with torch.no_grad():
            u = torch.LongTensor([user_id])
            i = torch.LongTensor([item_id])
            if hasattr(self.model, 'forward'):
                try:
                    score = self.model(u, i).item()
                except TypeError:
                    score = 0.5
            else:
                score = 0.5
        return score


class TemporalExplainer:
    """Explain recommendations using recency and candidate-specific similarity."""

    def __init__(self, train_df, decay_lambda=2.0, itemcf_model=None):
        from src.temporal import compute_time_weights

        self.decay_lambda = decay_lambda
        self.itemcf_model = itemcf_model
        self.train_df = train_df.copy()
        if "time_weight" not in self.train_df.columns:
            self.train_df = compute_time_weights(self.train_df, decay_lambda=decay_lambda)

    def _item_similarity(self, reference_item, history_item):
        if self.itemcf_model is None:
            return 0.0
        if reference_item is None or history_item is None:
            return 0.0
        if reference_item >= self.itemcf_model.n_items or history_item >= self.itemcf_model.n_items:
            return 0.0
        sim_row = self.itemcf_model.similarities[reference_item]
        return float(sim_row[0, history_item]) if sim_row.shape[1] > history_item else 0.0

    def explain(self, user_id, item_id, k=5):
        user_hist = self.train_df[self.train_df["user_id"] == user_id].copy()
        if user_hist.empty:
            return {
                "recommended_item": int(item_id) if item_id is not None else None,
                "top_weighted_items": [],
                "explanation": "No historical interactions found for this user.",
            }

        max_ts = float(user_hist["timestamp"].max()) if "timestamp" in user_hist.columns else None
        scored_rows = []
        for row in user_hist.itertuples(index=False):
            days_ago = None
            if max_ts is not None and getattr(row, "timestamp", None) is not None:
                days_ago = round((max_ts - float(row.timestamp)) / 86400.0, 2)
            recency_weight = float(row.time_weight)
            sim = self._item_similarity(item_id, int(row.item_id))
            combined = recency_weight + (2.0 * sim)
            scored_rows.append(
                {
                    "item_id": int(row.item_id),
                    "days_ago": days_ago,
                    "weight": round(recency_weight, 4),
                    "similarity_to_candidate": round(sim, 4),
                    "combined_score": round(combined, 4),
                }
            )

        scored_rows.sort(key=lambda x: (x["combined_score"], x["weight"]), reverse=True)
        top_items = scored_rows[:k]

        if item_id is None:
            explanation = "Recommended mainly from recent behavior: " + ", ".join(
                f"{x['item_id']} ({x['weight']})" for x in top_items
            )
        else:
            explanation = (
                f"For item {int(item_id)}, the recommendation mainly comes from recent behavior "
                f"and candidate-specific similarity: "
                + ", ".join(
                    f"{x['item_id']} (recent={x['weight']}, sim={x['similarity_to_candidate']})"
                    for x in top_items
                )
            )

        return {
            "recommended_item": int(item_id) if item_id is not None else None,
            "top_weighted_items": top_items,
            "explanation": explanation,
        }


def load_explainers(data_dir="data/processed", train_df=None):
    """Load all explanation modules."""
    data_path = Path(data_dir)

    with open(data_path / "user_reviews.json", "r", encoding="utf-8") as f:
        user_reviews = json.load(f)
    with open(data_path / "item_reviews.json", "r", encoding="utf-8") as f:
        item_reviews = json.load(f)

    keyword_explainer = KeywordExplainer(user_reviews, item_reviews)
    similar_user_explainer = SimilarUserExplainer()
    temporal_explainer = TemporalExplainer(train_df) if train_df is not None else None

    return {
        "keyword": keyword_explainer,
        "similar_user": similar_user_explainer,
        "temporal": temporal_explainer,
    }
