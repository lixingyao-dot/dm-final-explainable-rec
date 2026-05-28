"""LightGCN — Lightweight Graph Convolutional Network for Recommendation.

Pure PyTorch + sparse matrix multiplication. No PyTorch Geometric required.
"""

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.plotting import plot_training_history


class BPRDataset(Dataset):
    """Dataset for BPR training; samples one random negative per __getitem__ (per-epoch diversity)."""

    def __init__(self, train_df, n_items, n_neg=1):
        self.pos_pairs = train_df[["user_id", "item_id"]].values
        self.n_items = n_items
        self.n_neg = n_neg
        self.user_pos_items = train_df.groupby("user_id")["item_id"].apply(set).to_dict()

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, idx):
        user, pos_item = self.pos_pairs[idx]
        neg_items = []
        for _ in range(self.n_neg):
            neg = np.random.randint(0, self.n_items)
            while neg in self.user_pos_items.get(user, set()):
                neg = np.random.randint(0, self.n_items)
            neg_items.append(neg)
        return (
            torch.LongTensor([user]),
            torch.LongTensor([pos_item]),
            torch.LongTensor(neg_items),
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

    def forward(self, adj_norm):
        """
        Forward pass: propagate embeddings on the normalized adjacency matrix.

        Args:
            adj_norm: scipy sparse matrix (n_users+n_items) x (n_users+n_items),
                      normalized as D^{-1/2} A D^{-1/2}

        Returns:
            user_emb_final, item_emb_final: (n_users, d), (n_items, d)
        """
        # Convert sparse matrix to torch sparse tensor
        adj_tensor = self._scipy_to_torch(adj_norm).to(self.user_emb.weight.device)

        # Stack user and item embeddings
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        emb_list = [all_emb]

        # Multi-layer propagation
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_tensor, all_emb)
            emb_list.append(all_emb)

        # Mean of all layers
        final_emb = torch.stack(emb_list, dim=0).mean(dim=0)
        user_emb_final, item_emb_final = torch.split(
            final_emb, [self.n_users, self.n_items]
        )
        return user_emb_final, item_emb_final

    def predict(self, user_emb, item_emb, user_ids, item_ids):
        """Predict scores for given user-item pairs."""
        u_emb = user_emb[user_ids]
        i_emb = item_emb[item_ids]
        return (u_emb * i_emb).sum(dim=-1)

    def recommend(self, user_id, n_items, k, exclude=None, adj_norm=None, device=None):
        """Generate top-k recommendations for a user."""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward(adj_norm)
            scores = torch.matmul(user_emb[user_id], item_emb.t()).cpu().numpy()

        if exclude:
            for item in exclude:
                scores[item] = -999

        top_items = np.argsort(scores)[::-1][:k]
        return top_items.tolist()

    @staticmethod
    def _scipy_to_torch(sp_matrix):
        """Convert scipy sparse matrix to torch sparse tensor."""
        sp_matrix = sp_matrix.tocoo()
        indices = torch.LongTensor(np.stack([sp_matrix.row, sp_matrix.col]))
        values = torch.FloatTensor(sp_matrix.data)
        return torch.sparse_coo_tensor(indices, values, sp_matrix.shape)


def build_adj_matrix(train_df, n_users, n_items):
    """Build normalized adjacency matrix for LightGCN.

    Returns D^{-1/2} A D^{-1/2} as a scipy sparse matrix.
    """
    # Build bipartite graph adjacency
    users = train_df["user_id"].values
    items = train_df["item_id"].values + n_users  # offset items

    # A = [[0, R], [R^T, 0]]
    n_total = n_users + n_items
    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])
    data = np.ones(len(rows))

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_total, n_total))

    # Normalize: D^{-1/2} A D^{-1/2}
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)

    adj_norm = d_mat @ adj @ d_mat
    return adj_norm


def bpr_loss(pos_scores, neg_scores):
    """BPR loss: -log(sigmoid(pos - neg))."""
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))


def train_lightgcn(model, train_df, val_df, adj_norm, config, device=None):
    """Train LightGCN with BPR loss and early stopping."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = BPRDataset(train_df, config["model"].get("n_items", train_df["item_id"].nunique()))
    loader = DataLoader(dataset, batch_size=config["model"]["batch_size"], shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["model"]["early_stop_patience"]
    best_state = None

    train_losses, lr_history = [], []

    epoch_iter = tqdm(range(config["model"]["epochs"]), desc="LightGCN Training", unit="epoch")
    for epoch in epoch_iter:
        model.train()
        total_loss = 0
        n_batches = 0

        batch_iter = tqdm(loader, desc=f"  Epoch {epoch + 1}", leave=False, unit="batch")
        for user_ids, pos_ids, neg_ids in batch_iter:
            user_ids = user_ids.squeeze().to(device)
            pos_ids = pos_ids.squeeze().to(device)
            neg_ids = neg_ids.squeeze().to(device)

            user_emb, item_emb = model(adj_norm)
            pos_scores = model.predict(user_emb, item_emb, user_ids, pos_ids)
            neg_scores = model.predict(user_emb, item_emb, user_ids, neg_ids)

            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / n_batches

        epoch_iter.set_postfix(bpr_loss=f"{avg_loss:.4f}")

        train_losses.append(avg_loss)
        lr_history.append(optimizer.param_groups[0]["lr"])

        scheduler.step(avg_loss)

        # Simple early stopping on training loss (val loss needs full forward pass)
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                current_lr = optimizer.param_groups[0]["lr"]
                if current_lr <= scheduler.min_lrs[0]:
                    tqdm.write(f"    Early stopping at epoch {epoch + 1}")
                    break

    if best_state:
        model.load_state_dict(best_state)
    plot_training_history(
        train_losses, [], lr_history,
        "outputs/plots/lightgcn_training.png"
    )
    return model
