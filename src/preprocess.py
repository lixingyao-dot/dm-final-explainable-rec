"""End-to-end data processing pipeline. Run: python src/preprocess.py"""

import csv
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG
from src.utils import (
    load_checkpoint, save_checkpoint, should_skip,
    set_seed, print_section,
    sample_train_negatives, clean_review_text,
    aggregate_review_texts, print_rating_distribution,
)


# ── Stage 1: Load raw data ──────────────────────────────────────────────

def _resolve_raw_file(raw_file: str) -> Path:
    """Resolve raw file path; accept .json or .json.gz."""
    path = Path(raw_file)
    if path.exists():
        return path
    # try alternate extension
    alt = path.with_suffix(".json") if path.suffix == ".gz" else path.with_suffix(".json.gz")
    if alt.exists():
        return alt
    gz_stem = Path(str(path).replace(".json", ".json.gz"))
    if gz_stem.exists():
        return gz_stem
    return path


def _open_jsonl(path: Path):
    """Open JSONL file, supporting plain .json and .json.gz."""
    if path.suffix == ".gz" or str(path).endswith(".json.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _audit_json_fields(raw_file: Path, sample_lines: int = 10000) -> dict:
    """Scan sample lines to discover all JSON keys and their types."""
    key_counts = {}
    with _open_jsonl(raw_file) as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            obj = json.loads(line.strip())
            for key, val in obj.items():
                if key not in key_counts:
                    key_counts[key] = {"count": 0, "types": set()}
                key_counts[key]["count"] += 1
                key_counts[key]["types"].add(type(val).__name__)
    return key_counts


def _is_valid_review(obj: dict, field_map: dict) -> bool:
    """Basic validity check applied during streaming load."""
    if not obj.get(field_map["user_id"]) or not obj.get(field_map["item_id"]):
        return False
    try:
        rating = float(obj.get(field_map["rating"], 0))
    except (TypeError, ValueError):
        return False
    if not (1 <= rating <= 5):
        return False
    text = (obj.get(field_map["review_text"]) or "").strip()
    if not text:
        return False
    try:
        ts = int(obj.get(field_map["timestamp"], 0))
    except (TypeError, ValueError):
        return False
    return ts > 0


def _count_ratings_from_csv(csv_path: Path, chunksize: int = 500_000) -> Counter:
    """Count rating column from existing raw_interactions.csv."""
    counts: Counter = Counter()
    for chunk in pd.read_csv(csv_path, usecols=["rating"], chunksize=chunksize):
        for rating, cnt in chunk["rating"].value_counts().items():
            counts[int(rating)] += int(cnt)
    return counts


def load_raw_data():
    """Load Electronics_5.json(.gz) → raw_interactions.csv"""
    stage = "stage1"
    out_dir = Path(CONFIG["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_interactions.csv"

    if should_skip(stage):
        print("  → cached, skipping")
        if out_path.exists():
            print_rating_distribution(_count_ratings_from_csv(out_path), name="raw (cached)")
        return

    raw_file = _resolve_raw_file(CONFIG["data"]["raw_file"])
    field_map = CONFIG["fields"]

    if not raw_file.exists():
        print(f"  ERROR: raw file not found: {raw_file}")
        print(f"  Download from: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/")
        print(f"  Place Electronics_5.json or Electronics_5.json.gz in data/raw/")
        sys.exit(1)

    print("  Auditing JSON fields (first 10k lines) ...")
    audit = _audit_json_fields(raw_file)
    for key in sorted(audit.keys()):
        types = ", ".join(sorted(audit[key]["types"]))
        print(f"    {key}: {audit[key]['count']} occurrences, types=[{types}]")

    used_raw = set(field_map.values())
    unused = sorted(set(audit.keys()) - used_raw)
    if unused:
        print(f"  Unused fields (by design): {unused}")

    print(f"  Loading {raw_file} (streaming) ...")
    out_path = out_dir / "raw_interactions.csv"
    columns = list(field_map.keys())
    text_keys = {"reviewText", "summary"}
    n_rows = 0
    n_skipped = 0
    rating_counts: Counter = Counter()

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=columns)
        writer.writeheader()
        with _open_jsonl(raw_file) as f:
            for line in f:
                obj = json.loads(line.strip())
                if not _is_valid_review(obj, field_map):
                    n_skipped += 1
                    continue
                row = {
                    col: obj.get(raw_key, "" if raw_key in text_keys else 0)
                    for col, raw_key in field_map.items()
                }
                writer.writerow(row)
                rating_counts[int(row["rating"])] += 1
                n_rows += 1
                if n_rows % 500_000 == 0:
                    print(f"    ... {n_rows:,} rows written")

    print(f"  [raw] total rows: {n_rows:,} (skipped {n_skipped:,} invalid)")
    print(f"  columns: {columns}")
    print_rating_distribution(rating_counts, total=n_rows, name="raw")
    save_checkpoint(stage)
    print(f"  [OK] Stage 1 done: {n_rows:,} rows saved")


# ── Stage 2: Clean + K-core filter ─────────────────────────────────────

def clean_and_filter():
    """Basic cleaning + iterative K-core filtering (memory-efficient)."""
    stage = "stage2"
    if should_skip(stage):
        print("  → cached, skipping")
        return

    out_dir = Path(CONFIG["data"]["output_dir"])
    raw_path = out_dir / "raw_interactions.csv"
    slim_cols = ["user_id", "item_id", "rating", "timestamp"]

    print("  Loading slim columns for K-core ...")
    df = pd.read_csv(raw_path, usecols=slim_cols)
    n_before = len(df)
    df["timestamp"] = df["timestamp"].astype(int)
    print(f"  Rows after Stage-1 filter: {n_before:,}")

    # Step 2: K-core filtering on slim frame
    k = CONFIG["filter"]["k_core"]
    print(f"  K-core filtering (k={k}) ...")
    for round_i in range(20):
        n_users_prev = df["user_id"].nunique()
        n_items_prev = df["item_id"].nunique()

        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df["user_id"].isin(valid_users)]

        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df["item_id"].isin(valid_items)]

        n_users = df["user_id"].nunique()
        n_items = df["item_id"].nunique()
        print(f"    Round {round_i + 1}: users={n_users:,}, items={n_items:,}, interactions={len(df):,}")

        if n_users == n_users_prev and n_items == n_items_prev:
            print(f"  Converged at round {round_i + 1}")
            break

    valid_users = set(df["user_id"].unique())
    valid_items = set(df["item_id"].unique())
    print(f"  Merging text columns in chunks ...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(raw_path, chunksize=500_000)):
        mask = chunk["user_id"].isin(valid_users) & chunk["item_id"].isin(valid_items)
        filtered = chunk[mask]
        if len(filtered) > 0:
            chunks.append(filtered)
        if (i + 1) % 5 == 0:
            print(f"    ... scanned {(i + 1) * 500_000:,} raw rows")

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    df.to_csv(out_dir / "cleaned_interactions.csv", index=False)
    save_checkpoint(stage)
    print(f"  [OK] Stage 2 done: {len(df):,} interactions, {df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")


# ── Stage 3: ID reindex + sampling ─────────────────────────────────────

def reindex_and_sample():
    """Map IDs to 0..N-1, then sample users and items."""
    stage = "stage3"
    if should_skip(stage):
        print("  → cached, skipping")
        return

    out_dir = Path(CONFIG["data"]["output_dir"])
    df = pd.read_csv(out_dir / "cleaned_interactions.csv")

    # Preserve original Amazon IDs for mapping files
    df["orig_user_id"] = df["user_id"].astype(str)
    df["orig_item_id"] = df["item_id"].astype(str)

    # Reindex user IDs
    unique_users = df["user_id"].unique()
    user_map = {uid: i for i, uid in enumerate(unique_users)}
    df["user_id"] = df["user_id"].map(user_map)

    # Reindex item IDs
    unique_items = df["item_id"].unique()
    item_map = {iid: i for i, iid in enumerate(unique_items)}
    df["item_id"] = df["item_id"].map(item_map)

    print(f"  After reindex: {len(unique_users)} users, {len(unique_items)} items")

    # Sample users: prefer most active (denser subgraph)
    n_users_target = CONFIG["filter"]["n_users_sample"]
    n_items_target = CONFIG["filter"]["n_items_target"]
    min_per_user = CONFIG["filter"].get("min_interactions_per_user", 5)

    if len(unique_users) > n_users_target:
        set_seed(CONFIG["seed"])
        user_activity = df.groupby("user_id").size()
        if CONFIG["filter"].get("sample_active_users", True):
            sampled_users = user_activity.nlargest(n_users_target).index.tolist()
            print(f"  Sampled top {n_users_target} most active users")
        else:
            sampled_users = np.random.choice(
                df["user_id"].unique(), n_users_target, replace=False
            )
            print(f"  Randomly sampled {n_users_target} users")
        df = df[df["user_id"].isin(sampled_users)]

    # Remove items with < 2 distinct users
    item_user_counts = df.groupby("item_id")["user_id"].nunique()
    valid_items = item_user_counts[item_user_counts >= 2].index
    df = df[df["item_id"].isin(valid_items)]

    # Cap items to top-N by interaction count within subgraph
    if df["item_id"].nunique() > n_items_target:
        item_popularity = df["item_id"].value_counts()
        top_items = item_popularity.head(n_items_target).index
        df = df[df["item_id"].isin(top_items)]
        print(f"  Kept top {n_items_target} items by popularity")

    # Drop users with too few interactions after item trim; re-filter orphan items
    for _ in range(5):
        user_counts = df.groupby("user_id").size()
        valid_users = user_counts[user_counts >= min_per_user].index
        n_before = len(df)
        df = df[df["user_id"].isin(valid_users)]
        item_user_counts = df.groupby("item_id")["user_id"].nunique()
        valid_items = item_user_counts[item_user_counts >= 2].index
        df = df[df["item_id"].isin(valid_items)]
        if len(df) == n_before:
            break
    print(f"  After user/item quality filter: {df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items, {len(df):,} interactions")

    # Re-reindex to ensure contiguous IDs
    unique_users_final = df["user_id"].unique()
    unique_items_final = df["item_id"].unique()
    user_map_final = {uid: i for i, uid in enumerate(sorted(unique_users_final))}
    item_map_final = {iid: i for i, iid in enumerate(sorted(unique_items_final))}
    df["user_id"] = df["user_id"].map(user_map_final)
    df["item_id"] = df["item_id"].map(item_map_final)

    # Build original Amazon ID → final ID maps
    user_map_out = {
        row["orig_user_id"]: int(row["user_id"])
        for _, row in df.drop_duplicates("orig_user_id").iterrows()
    }
    item_map_out = {
        row["orig_item_id"]: int(row["item_id"])
        for _, row in df.drop_duplicates("orig_item_id").iterrows()
    }

    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_interactions = len(df)
    sparsity = 1 - n_interactions / (n_users * n_items)
    print(f"  Final: {n_users} users × {n_items} items × {n_interactions} interactions (sparsity={sparsity:.4f})")

    output_cols = ["user_id", "item_id", "rating", "review_text", "summary", "timestamp"]
    df[output_cols].to_csv(out_dir / "interactions.csv", index=False)
    with open(out_dir / "user_map.json", "w") as f:
        json.dump(user_map_out, f)
    with open(out_dir / "item_map.json", "w") as f:
        json.dump(item_map_out, f)

    save_checkpoint(stage)
    print(f"  [OK] Stage 3 done")


# ── Stage 4: Leave-One-Out split ───────────────────────────────────────

def split_train_val_test():
    """Leave-One-Out by time: last→test, second-to-last→val, rest→train."""
    stage = "stage4"
    if should_skip(stage):
        print("  → cached, skipping")
        return

    out_dir = Path(CONFIG["data"]["output_dir"])
    df = pd.read_csv(out_dir / "interactions.csv")

    min_interactions = CONFIG["filter"]["min_items_per_user"]
    train_list, val_list, test_list = [], [], []
    skipped = 0

    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("timestamp")
        if len(group) >= min_interactions:
            train_list.append(group.iloc[:-2])
            val_list.append(group.iloc[-2:-1])
            test_list.append(group.iloc[-1:])
        else:
            skipped += 1

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Skipped: {skipped} users")

    # Validate: val/test users must be in train
    train_users = set(train_df["user_id"].unique())
    val_users = set(val_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())
    assert val_users.issubset(train_users), "Val users not in train!"
    assert test_users.issubset(train_users), "Test users not in train!"
    print(f"  Validation passed: all val/test users exist in train")

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    save_checkpoint(stage)
    print(f"  [OK] Stage 4 done")


# ── Stage 5: Negative sampling ─────────────────────────────────────────

def negative_sampling():
    """Generate negative samples for train set."""
    stage = "stage5"
    if should_skip(stage):
        print("  → cached, skipping")
        return

    out_dir = Path(CONFIG["data"]["output_dir"])
    train_df = pd.read_csv(out_dir / "train.csv")
    interactions = pd.read_csv(out_dir / "interactions.csv")

    n_items = int(interactions["item_id"].max()) + 1
    neg_ratio = CONFIG["negative_sampling"]["neg_ratio"]
    set_seed(CONFIG["seed"])

    train_neg_df = sample_train_negatives(
        train_df, n_items, neg_ratio=neg_ratio, seed=CONFIG["seed"]
    )

    n_pos = (train_neg_df["label"] == 1).sum()
    n_neg = (train_neg_df["label"] == 0).sum()
    print(f"  Positive: {n_pos} | Negative: {n_neg} | Ratio: 1:{n_neg / n_pos:.1f}")

    train_neg_df.to_csv(out_dir / "train_neg.csv", index=False)
    save_checkpoint(stage)
    print(f"  [OK] Stage 5 done")


# ── Stage 6: Compute stats ─────────────────────────────────────────────

def compute_stats():
    """Compute and save dataset statistics."""
    stage = "stage6"
    if should_skip(stage):
        print("  → cached, skipping")
        return

    out_dir = Path(CONFIG["data"]["output_dir"])
    interactions = pd.read_csv(out_dir / "interactions.csv")
    train = pd.read_csv(out_dir / "train.csv")
    val = pd.read_csv(out_dir / "val.csv")
    test = pd.read_csv(out_dir / "test.csv")

    n_users = interactions["user_id"].nunique()
    n_items = interactions["item_id"].nunique()
    n_interactions = len(interactions)
    sparsity = 1 - n_interactions / (n_users * n_items)

    stats = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(n_interactions),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "sparsity": round(sparsity, 4),
        "avg_items_per_user": round(n_interactions / n_users, 1),
        "avg_users_per_item": round(n_interactions / n_items, 1),
    }

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Stats: {json.dumps(stats, indent=2)}")
    save_checkpoint(stage)
    print(f"  [OK] Stage 6 done")


# ── Stage 7: Review text preprocessing ─────────────────────────────────

def preprocess_text():
    """Clean and aggregate review texts per user and per item."""
    stage = "stage7"
    if should_skip(stage):
        print("  → cached, skipping")
        return

    out_dir = Path(CONFIG["data"]["output_dir"])
    df = pd.read_csv(out_dir / "interactions.csv")

    print(f"  Cleaning {len(df)} review texts (HTML / prefixes / noise) ...")
    df["review_text"] = df["review_text"].apply(clean_review_text)
    df["summary"] = df["summary"].apply(clean_review_text)

    text_cfg = CONFIG["text"]
    max_words = text_cfg["max_tokens"] * text_cfg.get("max_words_multiplier", 4)
    max_words_per_review = text_cfg.get("max_words_per_review")
    truncate_mode = text_cfg.get("truncate_mode", "head_tail")
    review_sep = text_cfg.get("review_sep", " . ")

    # Aggregate per user (time-ordered, head+tail truncate)
    print(f"  Aggregating user review texts (mode={truncate_mode}, max_words={max_words}) ...")
    user_reviews = {}
    for user_id, group in df.groupby("user_id"):
        combined = aggregate_review_texts(
            group,
            max_words=max_words,
            max_words_per_review=max_words_per_review,
            truncate_mode=truncate_mode,
            review_sep=review_sep,
        )
        if combined:
            user_reviews[int(user_id)] = combined

    # Aggregate per item
    print(f"  Aggregating item review texts ...")
    item_reviews = {}
    for item_id, group in df.groupby("item_id"):
        combined = aggregate_review_texts(
            group,
            max_words=max_words,
            max_words_per_review=max_words_per_review,
            truncate_mode=truncate_mode,
            review_sep=review_sep,
        )
        if combined:
            item_reviews[int(item_id)] = combined

    with open(out_dir / "user_reviews.json", "w", encoding="utf-8") as f:
        json.dump(user_reviews, f, ensure_ascii=False)
    with open(out_dir / "item_reviews.json", "w", encoding="utf-8") as f:
        json.dump(item_reviews, f, ensure_ascii=False)

    print(f"  User reviews: {len(user_reviews)} | Item reviews: {len(item_reviews)}")
    save_checkpoint(stage)
    print(f"  [OK] Stage 7 done")


# ── Stage 8: SBERT encoding ────────────────────────────────────────────

def _apply_hf_hub_mirror():
    """Use HF mirror for model download (works without shell env vars)."""
    endpoint = CONFIG["text"].get("hf_hub_endpoint", "").strip()
    if not endpoint:
        return
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_ENDPOINT"] = endpoint
    print(f"  Using HuggingFace hub mirror: {endpoint}")


def encode_sbert():
    """Encode aggregated review texts using Sentence-BERT."""
    stage = "stage8"
    if should_skip(stage):
        print("  → cached, skipping")
        return

    if CONFIG["text"].get("skip_sbert", False):
        print("  → skip_sbert=True in config, skipping SBERT encoding")
        return

    out_dir = Path(CONFIG["data"]["output_dir"])

    with open(out_dir / "user_reviews.json", "r", encoding="utf-8") as f:
        user_reviews = json.load(f)
    with open(out_dir / "item_reviews.json", "r", encoding="utf-8") as f:
        item_reviews = json.load(f)

    _apply_hf_hub_mirror()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  WARNING: sentence-transformers not installed. Run: pip install sentence-transformers")
        print("  Stage 8 skipped. NCF+Review will not work until embeddings are generated.")
        return

    model_name = CONFIG["text"]["sbert_model"]
    batch_size = CONFIG["text"]["batch_size"]
    print(f"  Loading SBERT model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"  WARNING: Failed to load SBERT model ({e})")
        print("  Tip: set HF_ENDPOINT=https://hf-mirror.com if HuggingFace is unreachable.")
        print("  Stage 8 skipped. Re-run pipeline after model is available.")
        return

    # Encode users (sorted by ID)
    user_ids = sorted(user_reviews.keys(), key=int)
    user_texts = [user_reviews[uid] if user_reviews[uid] else "no review" for uid in user_ids]
    print(f"  Encoding {len(user_texts)} user texts ...")
    user_emb = model.encode(user_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    user_emb = np.array(user_emb, dtype=np.float32)

    # Encode items (sorted by ID)
    item_ids = sorted(item_reviews.keys(), key=int)
    item_texts = [item_reviews[iid] if item_reviews[iid] else "no review" for iid in item_ids]
    print(f"  Encoding {len(item_texts)} item texts ...")
    item_emb = model.encode(item_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    item_emb = np.array(item_emb, dtype=np.float32)

    # Validate
    assert not np.isnan(user_emb).any(), "user_emb contains NaN!"
    assert not np.isnan(item_emb).any(), "item_emb contains NaN!"
    print(f"  user_emb shape: {user_emb.shape} | item_emb shape: {item_emb.shape}")

    np.save(out_dir / "user_emb.npy", user_emb)
    np.save(out_dir / "item_emb.npy", item_emb)

    save_checkpoint(stage)
    print(f"  [OK] Stage 8 done")


# ── Main entry ──────────────────────────────────────────────────────────

def run_pipeline():
    """Run the complete data processing pipeline end-to-end."""
    set_seed(CONFIG["seed"])

    stages = [
        ("Stage 1: Load raw data", load_raw_data),
        ("Stage 2: Clean + K-core filter", clean_and_filter),
        ("Stage 3: Reindex + sample", reindex_and_sample),
        ("Stage 4: Leave-One-Out split", split_train_val_test),
        ("Stage 5: Negative sampling", negative_sampling),
        ("Stage 6: Compute stats", compute_stats),
        ("Stage 7: Text preprocessing", preprocess_text),
        ("Stage 8: SBERT encoding", encode_sbert),
    ]

    print_section("Data Processing Pipeline")
    t0 = time.time()

    for name, fn in stages:
        print_section(name)
        t1 = time.time()
        fn()
        elapsed = time.time() - t1
        print(f"  [{name}] took {elapsed:.1f}s")

    total = time.time() - t0
    _print_summary(total)


def _print_summary(total_seconds: float):
    """Print a human-readable summary of the pipeline results."""
    out_dir = Path(CONFIG["data"]["output_dir"])
    stats_path = out_dir / "stats.json"

    if not stats_path.exists():
        return

    with open(stats_path, "r") as f:
        stats = json.load(f)

    user_emb_shape = "N/A"
    item_emb_shape = "N/A"
    if (out_dir / "user_emb.npy").exists():
        user_emb_shape = str(np.load(out_dir / "user_emb.npy").shape)
    if (out_dir / "item_emb.npy").exists():
        item_emb_shape = str(np.load(out_dir / "item_emb.npy").shape)

    mins, secs = divmod(int(total_seconds), 60)

    lines = []
    lines.append("=" * 60)
    lines.append("  Pipeline Summary")
    lines.append("=" * 60)
    lines.append(f"  Total time: {mins} min {secs} sec")
    lines.append(f"  Output:     {out_dir}")
    lines.append("")
    lines.append(f"  {'Metric':<28} {'Value':>12}")
    lines.append(f"  {'-'*28} {'-'*12}")
    rows = [
        ("Users", stats["n_users"]),
        ("Items", stats["n_items"]),
        ("Interactions", stats["n_interactions"]),
        ("Sparsity", f"{stats['sparsity']:.2%}"),
        ("Avg items per user", stats["avg_items_per_user"]),
        ("Train interactions", stats["n_train"]),
        ("Val interactions", stats["n_val"]),
        ("Test interactions", stats["n_test"]),
        ("SBERT embedding dims", f"{user_emb_shape} / {item_emb_shape}"),
    ]
    for label, value in rows:
        lines.append(f"  {label:<28} {str(value):>12}")

    for line in lines:
        print(line)

    summary_path = out_dir / "pipeline_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Amazon Electronics data")
    parser.add_argument("--users", type=int, default=None,
                        help="Number of users to sample (default from config)")
    parser.add_argument("--items", type=int, default=None,
                        help="Number of items to keep (default from config)")
    args = parser.parse_args()

    if args.users:
        CONFIG["filter"]["n_users_sample"] = args.users
    if args.items:
        CONFIG["filter"]["n_items_target"] = args.items

    if args.users or args.items:
        print(f"  Override: n_users_sample={CONFIG['filter']['n_users_sample']}, "
              f"n_items_target={CONFIG['filter']['n_items_target']}")

    run_pipeline()
