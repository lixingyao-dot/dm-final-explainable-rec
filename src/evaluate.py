"""Evaluation metrics for recommendation systems."""

import numpy as np


def _build_candidate_set(test_items: set, train_items: set, n_items: int, n_negatives: int, rng: np.random.Generator):
    """Sample negative items for one user. Returns (candidates: list, relevant_items: set)."""
    pos = list(test_items)
    seen = train_items | test_items
    pool = [i for i in range(n_items) if i not in seen]
    neg = rng.choice(pool, size=min(n_negatives, len(pool)), replace=False).tolist()

    candidates = neg + pos
    rng.shuffle(candidates)
    return candidates, set(pos)


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Precision@K = |recommended ∩ relevant| / K"""
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Recall@K = |recommended ∩ relevant| / |relevant|"""
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """Hit Rate@K = 1 if any relevant item in top-K, else 0"""
    rec_k = recommended[:k]
    return 1.0 if any(item in relevant for item in rec_k) else 0.0


def average_precision(recommended: list, relevant: set, k: int) -> float:
    """Average Precision for a single user."""
    rec_k = recommended[:k]
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(rec_k):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(len(relevant), k) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalized Discounted Cumulative Gain@K."""
    rec_k = recommended[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(rec_k) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model, test_df, train_df, n_items, top_ks=(5, 10, 20)):
    """
    Evaluate a recommendation model on the test set.

    Args:
        model: must have method recommend(user_id, n_items, k) -> list of item_ids
        test_df: DataFrame with columns [user_id, item_id]
        train_df: DataFrame with columns [user_id, item_id]
        n_items: total number of items
        top_ks: list of K values to evaluate

    Returns:
        dict of {metric@k: value}
    """
    # Build train set per user (items to exclude during evaluation)
    train_items_per_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    results = {f"{metric}@{k}": [] for k in top_ks
               for metric in ["Precision", "Recall", "HitRate", "MAP", "NDCG"]}

    for user_id, group in test_df.groupby("user_id"):
        relevant = set(group["item_id"].tolist())
        train_items = train_items_per_user.get(user_id, set())

        max_k = max(top_ks)
        recommended = model.recommend(user_id, n_items, max_k, exclude=train_items)

        for k in top_ks:
            results[f"Precision@{k}"].append(precision_at_k(recommended, relevant, k))
            results[f"Recall@{k}"].append(recall_at_k(recommended, relevant, k))
            results[f"HitRate@{k}"].append(hit_rate_at_k(recommended, relevant, k))
            results[f"MAP@{k}"].append(average_precision(recommended, relevant, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(recommended, relevant, k))

    # Average over all users
    return {k: round(np.mean(v), 4) for k, v in results.items()}


def evaluate_model_sampled(model, test_df, train_df, n_items, n_negatives=99, top_ks=(5, 10, 20), seed=42):
    """
    Sampled evaluation: only rank within 1 positive + N random negatives per user.

    This avoids the optimistic-baseline problem of ranking every item.
    """
    rng = np.random.default_rng(seed)
    train_items_per_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    results = {f"{metric}@{k}": [] for k in top_ks
               for metric in ["Precision", "Recall", "HitRate", "MAP", "NDCG"]}

    for user_id, group in test_df.groupby("user_id"):
        relevant = set(group["item_id"].tolist())
        train_items = train_items_per_user.get(user_id, set())

        # Build candidate set: N negatives + test positives
        candidates, pos_set = _build_candidate_set(relevant, train_items, n_items, n_negatives, rng)

        # Score only the candidate items (fast path for models that support it)
        if hasattr(model, "score_items"):
            scores = model.score_items(user_id, candidates)
            order = np.argsort(scores)[::-1]
            ranked = [candidates[i] for i in order]
        else:
            # Fallback: full ranking → project onto candidate set
            full_recs = model.recommend(user_id, n_items, n_items, exclude=train_items)
            candidate_set = set(candidates)
            ranked = [item for item in full_recs if item in candidate_set]
            for item in candidates:
                if item not in candidate_set.intersection(ranked):
                    ranked.append(item)

        for k in top_ks:
            results[f"Precision@{k}"].append(precision_at_k(ranked, pos_set, k))
            results[f"Recall@{k}"].append(recall_at_k(ranked, pos_set, k))
            results[f"HitRate@{k}"].append(hit_rate_at_k(ranked, pos_set, k))
            results[f"MAP@{k}"].append(average_precision(ranked, pos_set, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(ranked, pos_set, k))

    return {k: round(np.mean(v), 4) for k, v in results.items()}


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n  ── {model_name} Evaluation ──")
    k_values = sorted(set(int(k.split("@")[1]) for k in metrics.keys()))
    metric_names = ["Precision", "Recall", "HitRate", "MAP", "NDCG"]

    header = f"  {'Metric':<12}" + "".join(f"{'@' + str(k):<10}" for k in k_values)
    print(header)
    print(f"  {'─' * (12 + 10 * len(k_values))}")

    for metric in metric_names:
        row = f"  {metric:<12}"
        for k in k_values:
            key = f"{metric}@{k}"
            val = metrics.get(key, 0)
            row += f"{val:<10.4f}"
        print(row)
