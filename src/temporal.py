"""Temporal decay: weight interactions by recency for training.

Recent interactions get higher weight in the loss, forcing the model
to prioritize recent behavior patterns over stale history.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def compute_time_weights(train_df, decay_lambda=2.0):
    """Add a 'time_weight' column to train_df based on recency.

    weight = exp(-lambda * (t_max - t) / (t_max - t_min))

    Most recent interaction → weight ≈ 1.0
    Oldest interaction → weight ≈ exp(-lambda) ≈ 0.14 (for lambda=2)
    """
    df = train_df.copy()
    t = df["timestamp"].values.astype(float)
    t_min, t_max = t.min(), t.max()
    span = max(t_max - t_min, 1.0)
    df["time_weight"] = np.exp(-decay_lambda * (t_max - t) / span)
    return df


class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df["user_id"].values)
        self.items = torch.LongTensor(df["item_id"].values)
        self.labels = torch.FloatTensor(df["label"].values)
        if "time_weight" in df.columns:
            self.weights = torch.FloatTensor(df["time_weight"].values)
        else:
            self.weights = torch.ones(len(df))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx], self.weights[idx]


def train_ncf_temporal(model, train_df, val_df, config, n_items, device=None, decay_lambda=2.0):
    """Train NCF with time-decay weighted BCE loss."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    base_seed = config["seed"]
    neg_ratio = config["negative_sampling"]["neg_ratio"]

    from src.utils import sample_train_negatives, ensure_binary_labels

    # Build time-weighted training positives
    train_w = compute_time_weights(train_df, decay_lambda)
    train_w["label"] = 1  # all original rows are positives
    train_pos = train_w[["user_id", "item_id"]].copy()

    val_df = ensure_binary_labels(val_df)
    val_pos_df = val_df[val_df["label"] == 1][["user_id", "item_id"]].copy()
    val_users_items = val_pos_df.groupby("user_id")["item_id"].apply(set).to_dict()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_hitrate = 0.0
    patience_counter = 0
    patience = config["model"]["early_stop_patience"]
    best_state = None

    train_losses, val_losses, lr_history, val_hitrates = [], [], [], []

    # Pre-sample validation candidates
    n_neg_val = 99
    val_rng = np.random.default_rng(base_seed + 7777)
    val_candidates = {}
    for uid, relevant_items in val_users_items.items():
        train_set = set(train_pos[train_pos["user_id"] == uid]["item_id"].values)
        pos = list(relevant_items)
        seen = train_set | relevant_items
        pool = [i for i in range(n_items) if i not in seen]
        neg = val_rng.choice(pool, size=min(n_neg_val, len(pool)), replace=False).tolist()
        candidates = neg + pos
        val_rng.shuffle(candidates)
        val_candidates[uid] = (candidates, relevant_items)

    epoch_iter = tqdm(range(config["model"]["epochs"]), desc="NCF Temporal Training", unit="epoch")
    for epoch in epoch_iter:
        # Resample negatives each epoch, positives keep time weights
        neg = sample_train_negatives(train_pos, n_items, neg_ratio=neg_ratio, seed=base_seed + epoch)
        neg_pos = neg[neg["label"] == 1]
        neg_neg = neg[neg["label"] == 0]

        # Combine: weighted positives + unweighted negatives
        train_data = train_w[["user_id", "item_id", "label", "time_weight"]].copy()
        neg_rows = pd.DataFrame({
            "user_id": neg_neg["user_id"],
            "item_id": neg_neg["item_id"],
            "label": 0,
            "time_weight": 1.0,  # negatives use full weight
        })
        train_data = pd.concat([train_data, neg_rows], ignore_index=True)
        train_data = train_data.sample(frac=1, random_state=base_seed + epoch).reset_index(drop=True)

        loader = DataLoader(
            InteractionDataset(train_data),
            batch_size=config["model"]["batch_size"],
            shuffle=True,
        )

        model.train()
        total_loss = 0
        n_batches = 0

        batch_iter = tqdm(loader, desc=f"  Epoch {epoch + 1}", leave=False, unit="batch")
        for users, items, labels, weights in batch_iter:
            users, items = users.to(device), items.to(device)
            labels, weights = labels.to(device), weights.to(device)

            optimizer.zero_grad()
            preds = model(users, items)
            # Weighted BCE: recent positives contribute more to the loss
            bce = nn.functional.binary_cross_entropy(preds, labels, reduction="none")
            loss = (bce * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / n_batches

        # Validation loss (unweighted BCE)
        model.eval()
        val_loss = 0
        val_n = 0
        val_loader = DataLoader(
            InteractionDataset(pd.concat([neg_pos, neg_neg])),
            batch_size=config["model"]["batch_size"]
        )
        with torch.no_grad():
            for users, items, labels, weights in val_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                preds = model(users, items)
                loss = nn.functional.binary_cross_entropy(preds, labels)
                val_loss += loss.item() * len(labels)
                val_n += len(labels)
        val_loss = val_loss / val_n

        # Sampled HitRate@10
        val_hitrate = 0.0
        val_k = 10
        for uid, (candidates, relevant_items) in val_candidates.items():
            scores = model.score_items(uid, candidates, device=device)
            top = [candidates[i] for i in np.argsort(scores)[::-1][:val_k]]
            if any(item in relevant_items for item in top):
                val_hitrate += 1.0
        val_hitrate /= len(val_candidates) if val_candidates else 1.0

        epoch_iter.set_postfix(
            train_loss=f"{avg_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            hitrate=f"{val_hitrate:.4f}"
        )

        train_losses.append(avg_loss)
        val_losses.append(val_loss)
        lr_history.append(optimizer.param_groups[0]["lr"])
        val_hitrates.append(val_hitrate)
        scheduler.step(val_loss)

        if val_hitrate > best_val_hitrate:
            best_val_hitrate = val_hitrate
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(f"    Early stopping at epoch {epoch + 1} (best HitRate@10={best_val_hitrate:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    from src.plotting import plot_training_history
    plot_training_history(
        train_losses, val_losses, lr_history,
        f"outputs/plots/ncf_temporal_l{decay_lambda}.png",
        val_hitrates=val_hitrates
    )
    return model
