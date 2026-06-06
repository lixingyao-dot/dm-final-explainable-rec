"""Batch explanation export for item-based and temporal explanations."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explain import TemporalExplainer
from src.base_model.itemcf import ItemCF
from src.model_loading import load_best_recommender


def _load_processed_data(data_dir: Path):
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    return train_df, test_df, stats


def _rank_user_items(recommender, user_id, n_items, exclude=None, top_k=10):
    """Return top-k items with scores for one user."""
    exclude = exclude or set()
    candidate_items = [i for i in range(n_items) if i not in exclude]
    if not candidate_items:
        return []
    scores = recommender.score_items(int(user_id), candidate_items)
    order = scores.argsort()[::-1][:top_k]
    return [
        {
            "item_id": int(candidate_items[idx]),
            "score": float(scores[idx]),
        }
        for idx in order
    ]


def build_explanations(
    data_dir="data/processed",
    model_dir="outputs/models",
    output_path="outputs/exp6_explanations.json",
    max_users=None,
):
    """Generate recommendation explanations for users in the processed test set."""
    data_path = Path(data_dir)
    model_path = Path(model_dir)
    output_path = Path(output_path)

    train_df, test_df, stats = _load_processed_data(data_path)
    n_users = int(stats["n_users"])
    n_items = int(stats["n_items"])

    recommender, recommender_ckpt = load_best_recommender(model_path)
    itemcf = ItemCF(train_df, n_users=n_users, n_items=n_items)
    temporal_explainer = TemporalExplainer(train_df, itemcf_model=itemcf)

    train_items_per_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    results = []

    for idx, (user_id, group) in enumerate(test_df.groupby("user_id")):
        if max_users is not None and idx >= max_users:
            break

        relevant_items = group["item_id"].tolist()
        user_exclude = train_items_per_user.get(user_id, set())
        recommendations = _rank_user_items(
            recommender,
            int(user_id),
            n_items=n_items,
            exclude=user_exclude,
            top_k=10,
        )
        top_rec = recommendations[0] if recommendations else None
        recommended_item = int(top_rec["item_id"]) if top_rec else None
        recommended_score = float(top_rec["score"]) if top_rec else None

        itemcf_explanation = itemcf.explain(int(user_id), recommended_item, k=5) if recommended_item is not None else {
            "recommended_item": None,
            "bridge_items": [],
            "explanation": "No recommendation available.",
        }
        temporal_explanation = temporal_explainer.explain(int(user_id), recommended_item, k=5) if recommended_item is not None else {
            "recommended_item": None,
            "top_weighted_items": [],
            "explanation": "No recommendation available.",
        }

        results.append(
            {
                "user_id": int(user_id),
                "relevant_items": [int(x) for x in relevant_items],
                "recommended_item": recommended_item,
                "recommended_score": recommended_score,
                "recommendations": recommendations,
                "itemcf_explanation": itemcf_explanation,
                "temporal_explanation": temporal_explanation,
            }
        )

    payload = {
        "n_users": len(results),
        "total_users_available": int(test_df["user_id"].nunique()),
        "n_items": n_items,
        "model_dir": str(model_path),
        "recommender_checkpoint": str(recommender_ckpt),
        "recommender_class": recommender.__class__.__name__,
        "recommendations": results[0]["recommendations"] if results else [],
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "n_users": len(results),
        "n_items": n_items,
        "output_path": str(output_path),
        "model_dir": str(model_path),
        "recommender_checkpoint": str(recommender_ckpt),
        "recommender_class": recommender.__class__.__name__,
        "recommendations": results[0]["recommendations"] if results else [],
    }


def main():
    result = build_explanations()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
