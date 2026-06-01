"""NCF trained with BPR (Bayesian Personalized Ranking) loss.

Key difference from vanilla NCF:
- BCE loss → BPR loss (pairwise ranking objective)
- Training: (user, pos_item, neg_item) triples, maximize pos - neg score gap
- More aligned with recommendation's ranking goal than pointwise BCE
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.utils import ensure_binary_labels
from src.plotting import plot_training_history


class BPRDataset(Dataset):
    """Each sample: (user_id, pos_item_id, neg_item_id)."""

    def __init__(self, user_pos_items, n_items, neg_per_pos=1, seed=42):
        """
        user_pos_items: dict {user_id: list of positive item_ids}
        """
        self.n_items = n_items
        all_items = set(range(n_items))
        self.triples = []
        rng = np.random.default_rng(seed)

        for uid, pos_list in user_pos_items.items():
            pos_set = set(pos_list)
            neg_pool = list(all_items - pos_set)
            for pos in pos_list:
                for _ in range(neg_per_pos):
                    neg = int(rng.choice(neg_pool))
                    self.triples.append((uid, pos, neg))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        u, i, j = self.triples[idx]
        return (
            torch.LongTensor([u])[0],
            torch.LongTensor([i])[0],
            torch.LongTensor([j])[0],
        )


class NCFBPR(nn.Module):
    """NCF model for BPR training. Architecture same as vanilla NCF (GMF+MLP fusion).
    No sigmoid on output — BPR uses raw logits for ranking."""

    def __init__(self, n_users, n_items, embedding_dim=64, mlp_layers=(64, 32, 16)):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_emb_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_emb_gmf = nn.Embedding(n_items, embedding_dim)
        self.user_emb_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_emb_mlp = nn.Embedding(n_items, embedding_dim)

        mlp_input_dim = embedding_dim * 2
        layers = []
        for dim in mlp_layers:
            layers.append(nn.Linear(mlp_input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            mlp_input_dim = dim
        self.mlp = nn.Sequential(*layers)

        self.output_layer = nn.Linear(mlp_layers[-1] + embedding_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        u_gmf = self.user_emb_gmf(user_ids)
        i_gmf = self.item_emb_gmf(item_ids)
        gmf_out = u_gmf * i_gmf

        u_mlp = self.user_emb_mlp(user_ids)
        i_mlp = self.item_emb_mlp(item_ids)
        mlp_input = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_input)

        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        score = self.output_layer(concat)
        return score.squeeze(-1)  # raw logit, no sigmoid

    def score_items(self, user_id, items, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * len(items)).to(device)
            item_tensor = torch.LongTensor(items).to(device)
            scores = self.forward(user_tensor, item_tensor).cpu().numpy()
        return scores

    def recommend(self, user_id, n_items, k, exclude=None, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * n_items).to(device)
            item_tensor = torch.LongTensor(list(range(n_items))).to(device)
            scores = self.forward(user_tensor, item_tensor).cpu().numpy()

        if exclude:
            for item in exclude:
                scores[item] = -999

        top_items = np.argsort(scores)[::-1][:k]
        return top_items.tolist()


def bpr_loss_fn(model, users, pos_items, neg_items):
    """BPR loss: -log(sigmoid(pos_score - neg_score)) + L2 reg."""
    pos_scores = model(users, pos_items)
    neg_scores = model(users, neg_items)
    diff = pos_scores - neg_scores
    # L2 regularization on embeddings
    reg = 0.0
    for name, param in model.named_parameters():
        if "emb" in name:
            reg += torch.norm(param) * 1e-5
    return -torch.log(torch.sigmoid(diff) + 1e-10).mean() + reg


def train_ncf_bpr(model, train_pos_df, val_df, config, n_items, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    base_seed = config["seed"]

    if "label" in train_pos_df.columns:
        train_pos_df = train_pos_df[train_pos_df["label"] == 1][["user_id", "item_id"]]
    else:
        train_pos_df = train_pos_df[["user_id", "item_id"]].copy()

    # Build user → pos_items dict
    user_pos_items = train_pos_df.groupby("user_id")["item_id"].apply(list).to_dict()

    val_df = ensure_binary_labels(val_df)
    val_users_items = (
        val_df[val_df["label"] == 1].groupby("user_id")["item_id"].apply(set).to_dict()
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["model"]["learning_rate"], weight_decay=0
    )

    best_val_hitrate = 0.0
    patience_counter = 0
    patience = config["model"]["early_stop_patience"]
    best_state = None

    train_losses, val_hitrates = [], []

    # Pre-sample validation candidates
    n_neg = 99
    val_rng = np.random.default_rng(base_seed + 7777)
    val_candidates = {}
    for uid, relevant_items in val_users_items.items():
        train_set = set(train_pos_df[train_pos_df["user_id"] == uid]["item_id"].values)
        pos = list(relevant_items)
        seen = train_set | relevant_items
        pool = [i for i in range(n_items) if i not in seen]
        neg = val_rng.choice(pool, size=min(n_neg, len(pool)), replace=False).tolist()
        candidates = neg + pos
        val_rng.shuffle(candidates)
        val_candidates[uid] = (candidates, relevant_items)

    epoch_iter = tqdm(range(config["model"]["epochs"]), desc="NCF+BPR Training", unit="epoch")
    neg_per_pos = config["negative_sampling"]["neg_ratio"]  # use same ratio as config (e.g. 9)

    for epoch in epoch_iter:
        # Resample negative items every epoch with varying seed
        train_dataset = BPRDataset(user_pos_items, n_items, neg_per_pos=neg_per_pos,
                                   seed=base_seed + epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=config["model"]["batch_size"],
            shuffle=True,
        )

        model.train()
        total_loss = 0
        n_batches = 0

        batch_iter = tqdm(loader, desc=f"  Epoch {epoch + 1}", leave=False, unit="batch")
        for users, pos_items, neg_items in batch_iter:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            optimizer.zero_grad()
            loss = bpr_loss_fn(model, users, pos_items, neg_items)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / n_batches

        # Sampled HitRate@10 validation
        model.eval()
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
            hitrate=f"{val_hitrate:.4f}"
        )

        train_losses.append(avg_loss)
        val_hitrates.append(val_hitrate)

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
    n = len(train_losses)
    plot_training_history(
        train_losses, [0.0] * n, [0.0] * n,
        "outputs/plots/ncf_bpr_training.png",
        val_hitrates=val_hitrates
    )
    return model
