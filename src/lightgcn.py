"""LightGCN — Lightweight Graph Convolutional Network for Recommendation.

Fixed:
- Full graph propagation once per epoch (not per batch)
- Sampled validation HitRate after each epoch
- Proper early stopping on validation metric
- score_items() for sampled evaluation
- Deterministic negative sampling per epoch
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.sparse as sp
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.plotting import plot_training_history


class BPRTripletDataset(torch.utils.data.Dataset):
    """Yield (user, pos_item, neg_item) triples, pre-sampled for one epoch."""

    def __init__(self, user_pos_items, n_items, neg_per_pos=1, seed=42):
        rng = np.random.default_rng(seed)
        all_items = set(range(n_items))
        self.triples = []
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


class LightGCN(nn.Module):
    """LightGCN model: graph convolution on user-item bipartite graph."""

    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def propagate(self, adj_tensor):
        """Full graph propagation: returns (user_emb_final, item_emb_final)."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_tensor, all_emb)
            emb_list.append(all_emb)
        final_emb = torch.stack(emb_list, dim=0).mean(dim=0)
        user_emb, item_emb = torch.split(final_emb, [self.n_users, self.n_items])
        return user_emb, item_emb

    def predict(self, user_emb, item_emb, user_ids, item_ids):
        """Inner product of user and item embeddings."""
        return (user_emb[user_ids] * item_emb[item_ids]).sum(dim=-1)

    def score_items(self, user_id, items, adj_tensor, device=None):
        """Score a specific list of items for a user."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.propagate(adj_tensor)
            u = user_emb[user_id]
            i = item_emb[torch.LongTensor(items).to(device)]
            return (u * i).sum(dim=-1).cpu().numpy()

    def recommend(self, user_id, n_items, k, exclude=None, adj_norm=None, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        adj_tensor = _scipy_to_torch(adj_norm).to(device)
        with torch.no_grad():
            user_emb, item_emb = self.propagate(adj_tensor)
            scores = torch.matmul(user_emb[user_id], item_emb.t()).cpu().numpy()

        if exclude:
            for item in exclude:
                scores[item] = -999

        top_items = np.argsort(scores)[::-1][:k]
        return top_items.tolist()


def _scipy_to_torch(sp_matrix):
    """Convert scipy CSR matrix to torch sparse COO tensor."""
    sp_matrix = sp_matrix.tocoo()
    indices = torch.LongTensor(np.stack([sp_matrix.row, sp_matrix.col]))
    values = torch.FloatTensor(sp_matrix.data)
    return torch.sparse_coo_tensor(indices, values, sp_matrix.shape)


def build_adj_matrix(train_df, n_users, n_items):
    """Build normalized adjacency matrix D^{-1/2} A D^{-1/2}."""
    users = train_df["user_id"].values
    items = train_df["item_id"].values + n_users
    n_total = n_users + n_items
    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])
    data = np.ones(len(rows))
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_total, n_total))

    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return d_mat @ adj @ d_mat


def bpr_loss_fn(pos_scores, neg_scores):
    """BPR loss: -log(sigmoid(pos - neg)) + L2 on embeddings."""
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))


def train_lightgcn(model, train_pos_df, val_df, adj_norm, config, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    base_seed = config["seed"]

    if "label" in train_pos_df.columns:
        train_pos_df = train_pos_df[train_pos_df["label"] == 1][["user_id", "item_id"]]
    else:
        train_pos_df = train_pos_df[["user_id", "item_id"]].copy()

    user_pos_items = train_pos_df.groupby("user_id")["item_id"].apply(list).to_dict()
    n_users = model.n_users
    n_items = model.n_items

    # Pre-build torch sparse tensor for the adjacency matrix (reused every epoch)
    adj_tensor = _scipy_to_torch(adj_norm).to(device)

    # Validation setup
    from src.utils import ensure_binary_labels
    val_df = ensure_binary_labels(val_df)
    val_users_items = (
        val_df[val_df["label"] == 1].groupby("user_id")["item_id"].apply(set).to_dict()
    )

    n_neg_val = 99
    val_rng = np.random.default_rng(base_seed + 7777)
    val_candidates = {}
    for uid, relevant_items in val_users_items.items():
        train_set = set(train_pos_df[train_pos_df["user_id"] == uid]["item_id"].values)
        pos = list(relevant_items)
        seen = train_set | relevant_items
        pool = [i for i in range(n_items) if i not in seen]
        neg = val_rng.choice(pool, size=min(n_neg_val, len(pool)), replace=False).tolist()
        candidates = neg + pos
        val_rng.shuffle(candidates)
        val_candidates[uid] = (candidates, relevant_items)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-5
    )

    best_val_hitrate = 0.0
    patience_counter = 0
    patience = config["model"]["early_stop_patience"]
    best_state = None

    train_losses, val_hitrates = [], []
    neg_per_pos = config["negative_sampling"]["neg_ratio"]

    epoch_iter = tqdm(range(config["model"]["epochs"]), desc="LightGCN Training", unit="epoch")
    for epoch in epoch_iter:
        # 1. Propagate ONCE per epoch
        model.train()
        user_emb, item_emb = model.propagate(adj_tensor)

        # 2. Resample negatives for this epoch
        train_dataset = BPRTripletDataset(user_pos_items, n_items,
                                          neg_per_pos=neg_per_pos,
                                          seed=base_seed + epoch)
        loader = DataLoader(train_dataset, batch_size=config["model"]["batch_size"], shuffle=True)

        total_loss = 0
        n_batches = 0

        batch_iter = tqdm(loader, desc=f"  Epoch {epoch + 1}", leave=False, unit="batch")
        for users, pos_items, neg_items in batch_iter:
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            pos_scores = model.predict(user_emb, item_emb, users, pos_items)
            neg_scores = model.predict(user_emb, item_emb, users, neg_items)
            loss = bpr_loss_fn(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / n_batches

        # 3. Validation: sampled HitRate@10
        model.eval()
        val_hitrate = 0.0
        val_k = 10
        with torch.no_grad():
            user_emb_val, item_emb_val = model.propagate(adj_tensor)
            for uid, (candidates, relevant_items) in val_candidates.items():
                u_vec = user_emb_val[uid]
                i_vecs = item_emb_val[torch.LongTensor(candidates).to(device)]
                scores = (u_vec * i_vecs).sum(dim=-1).cpu().numpy()
                top = [candidates[i] for i in np.argsort(scores)[::-1][:val_k]]
                if any(item in relevant_items for item in top):
                    val_hitrate += 1.0
        val_hitrate /= len(val_candidates) if val_candidates else 1.0

        epoch_iter.set_postfix(
            bpr_loss=f"{avg_loss:.4f}",
            hitrate=f"{val_hitrate:.4f}"
        )

        train_losses.append(avg_loss)
        val_hitrates.append(val_hitrate)

        # 4. Early stopping on validation HitRate
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
        "outputs/plots/lightgcn_training.png",
        val_hitrates=val_hitrates
    )
    return model
