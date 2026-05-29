"""NCF with user/item bias terms.

Key improvement over vanilla NCF:
- user_bias: scalar per user, captures tendency to interact with many items
- item_bias: scalar per item, captures popularity signal (what ItemCF exploits)
- score = GMF_out + MLP_out + user_bias + item_bias
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import sample_train_negatives, ensure_binary_labels
from src.plotting import plot_training_history


class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df["user_id"].values)
        self.items = torch.LongTensor(df["item_id"].values)
        self.labels = torch.FloatTensor(df["label"].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


class NCFBias(nn.Module):
    """NCF + user/item bias. Same GMF+MLP fusion, plus learnable scalar biases."""

    def __init__(self, n_users, n_items, embedding_dim=32, mlp_layers=(64, 32, 16)):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # GMF path
        self.user_emb_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_emb_gmf = nn.Embedding(n_items, embedding_dim)

        # MLP path
        self.user_emb_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_emb_mlp = nn.Embedding(n_items, embedding_dim)

        # Bias terms — explicitly capture popularity signal
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # MLP layers
        mlp_input_dim = embedding_dim * 2
        layers = []
        for dim in mlp_layers:
            layers.append(nn.Linear(mlp_input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            mlp_input_dim = dim
        self.mlp = nn.Sequential(*layers)

        # Final prediction: GMF_out + MLP_out + biases → score
        self.output_layer = nn.Linear(mlp_layers[-1] + embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        # GMF
        u_gmf = self.user_emb_gmf(user_ids)
        i_gmf = self.item_emb_gmf(item_ids)
        gmf_out = u_gmf * i_gmf

        # MLP
        u_mlp = self.user_emb_mlp(user_ids)
        i_mlp = self.item_emb_mlp(item_ids)
        mlp_input = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Concat + bias
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        score = self.output_layer(concat) + self.user_bias(user_ids) + self.item_bias(item_ids) + self.global_bias
        return self.sigmoid(score).squeeze(-1)

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


def train_ncf_bias(model, train_pos_df, val_df, config, n_items, device=None):
    """Train NCFBias with sampled validation and early stopping."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    neg_ratio = config["negative_sampling"]["neg_ratio"]
    resample = config["negative_sampling"].get("resample_per_epoch", True)
    base_seed = config["seed"]

    if "label" in train_pos_df.columns:
        train_pos_df = train_pos_df[train_pos_df["label"] == 1][["user_id", "item_id"]]
    else:
        train_pos_df = train_pos_df[["user_id", "item_id"]].copy()

    val_df = ensure_binary_labels(val_df)
    val_neg = sample_train_negatives(
        val_df[val_df["label"] == 1][["user_id", "item_id"]],
        n_items, neg_ratio=neg_ratio, seed=base_seed + 9999
    )
    val_df = val_df[val_df["label"] == 1][["user_id", "item_id"]].copy()
    val_df["label"] = 1
    val_df = pd.concat([val_df, val_neg[val_neg["label"] == 0]], ignore_index=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-5
    )
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_hitrate = 0.0
    patience_counter = 0
    patience = config["model"]["early_stop_patience"]
    best_state = None

    train_losses, val_losses, lr_history, val_hitrates = [], [], [], []

    val_pos_df = val_df[val_df["label"] == 1][["user_id", "item_id"]].copy()
    val_users_items = val_pos_df.groupby("user_id")["item_id"].apply(set).to_dict()

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

    if not resample:
        static_train = sample_train_negatives(
            train_pos_df, n_items, neg_ratio=neg_ratio, seed=base_seed
        )

    epoch_iter = tqdm(range(config["model"]["epochs"]), desc="NCFBias Training", unit="epoch")
    for epoch in epoch_iter:
        if resample:
            train_data = sample_train_negatives(
                train_pos_df, n_items, neg_ratio=neg_ratio, seed=base_seed + epoch
            )
        else:
            train_data = static_train

        loader = DataLoader(
            InteractionDataset(train_data),
            batch_size=config["model"]["batch_size"],
            shuffle=True,
        )

        model.train()
        total_loss = 0
        n_batches = 0

        batch_iter = tqdm(loader, desc=f"  Epoch {epoch + 1}", leave=False, unit="batch")
        for users, items, labels in batch_iter:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0
        val_n = 0
        val_dataset = InteractionDataset(val_df)
        val_loader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"])
        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                preds = model(users, items)
                loss = criterion(preds, labels)
                val_loss += loss.item() * len(labels)
                val_n += len(labels)
        val_loss /= val_n

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
    plot_training_history(
        train_losses, val_losses, lr_history,
        "outputs/plots/ncf_bias_training.png",
        val_hitrates=val_hitrates
    )
    return model
