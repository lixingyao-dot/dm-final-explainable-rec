"""NCF + Review Embedding Fusion Model.

Fuses behavioral embeddings (from NCF) with semantic embeddings (from SBERT).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.utils import sample_train_negatives, ensure_binary_labels
from src.plotting import plot_training_history


class NCFReviewDataset(Dataset):
    def __init__(self, df, user_emb, item_emb):
        self.users = torch.LongTensor(df["user_id"].values)
        self.items = torch.LongTensor(df["item_id"].values)
        self.labels = torch.FloatTensor(df["label"].values)
        self.user_review_emb = torch.FloatTensor(user_emb)
        self.item_review_emb = torch.FloatTensor(item_emb)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]
        return (
            u, i,
            self.user_review_emb[u],
            self.item_review_emb[i],
            self.labels[idx],
        )


class NCFReview(nn.Module):
    """NCF + Review Embedding fusion model.

    Architecture:
    1. NCF path: user/item ID embeddings -> MLP -> behavior_vec
    2. SBERT path: user/item review embeddings -> linear projection -> semantic_vec
    3. Weighted fusion: final = (1-alpha) * behavior + alpha * semantic
    4. Prediction: final -> MLP -> score
    """

    def __init__(self, n_users, n_items, embedding_dim=64,
                 mlp_layers=(64, 32, 16), review_emb_dim=384, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.embedding_dim = embedding_dim
        self.behavior_output_dim = mlp_layers[-1]

        # NCF embeddings
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)

        # Review embedding projection: 384 -> behavior_output_dim
        self.user_review_proj = nn.Linear(review_emb_dim, self.behavior_output_dim)
        self.item_review_proj = nn.Linear(review_emb_dim, self.behavior_output_dim)

        # MLP for behavior path
        behavior_input_dim = embedding_dim * 2
        behavior_layers = []
        for dim in mlp_layers:
            behavior_layers.extend([
                nn.Linear(behavior_input_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            behavior_input_dim = dim
        self.behavior_mlp = nn.Sequential(*behavior_layers)

        # Fusion MLP
        fusion_input_dim = self.behavior_output_dim
        fusion_layers = [
            nn.Linear(fusion_input_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid(),
        ]
        self.fusion_mlp = nn.Sequential(*fusion_layers)

        # Cached review embeddings for evaluation (set via set_review_embeddings)
        self.register_buffer('_cached_user_review_emb', None)
        self.register_buffer('_cached_item_review_emb', None)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def set_review_embeddings(self, user_emb, item_emb):
        """Store review embeddings for later evaluation (used by evaluate_model)."""
        self._cached_user_review_emb = torch.FloatTensor(user_emb)
        self._cached_item_review_emb = torch.FloatTensor(item_emb)

    def forward(self, user_ids, item_ids, user_review_emb, item_review_emb):
        # Behavior path
        u_beh = self.user_emb(user_ids)
        i_beh = self.item_emb(item_ids)
        behavior_input = torch.cat([u_beh, i_beh], dim=-1)
        behavior_vec = self.behavior_mlp(behavior_input)

        # Semantic path
        u_sem = self.user_review_proj(user_review_emb)
        i_sem = self.item_review_proj(item_review_emb)
        semantic_vec = u_sem + i_sem

        # Weighted fusion
        final_vec = (1 - self.alpha) * behavior_vec + self.alpha * semantic_vec

        # Predict
        score = self.fusion_mlp(final_vec)
        return score.squeeze(-1)

    def recommend(self, user_id, n_items, k, exclude=None,
                  user_emb_tensor=None, item_emb_tensor=None, device=None):
        """Generate top-k recommendations for a user.

        Falls back to cached embeddings (set via set_review_embeddings)
        if no tensor arguments passed (compatible with evaluate_model).
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * n_items).to(device)
            item_tensor = torch.LongTensor(list(range(n_items))).to(device)

            # Use provided tensors or cached ones
            u_rev_src = user_emb_tensor if user_emb_tensor is not None else self._cached_user_review_emb
            i_rev_src = item_emb_tensor if item_emb_tensor is not None else self._cached_item_review_emb

            if u_rev_src is None or i_rev_src is None:
                raise RuntimeError(
                    "NCFReview.recommend requires review embeddings. "
                    "Call set_review_embeddings(user_emb, item_emb) first."
                )

            u_rev = u_rev_src[user_id].unsqueeze(0).expand(n_items, -1).to(device)
            i_rev = i_rev_src.to(device)

            scores = self.forward(user_tensor, item_tensor, u_rev, i_rev).cpu().numpy()

        if exclude:
            for item in exclude:
                scores[item] = -999

        top_items = np.argsort(scores)[::-1][:k]
        return top_items.tolist()


def train_ncf_review(model, train_pos_df, val_df, user_emb, item_emb, config, n_items, device=None):
    """Train NCF+Review fusion model with validation-based early stopping."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    neg_ratio = config["negative_sampling"]["neg_ratio"]
    resample = config["negative_sampling"].get("resample_per_epoch", True)
    base_seed = config["seed"]

    if "label" in train_pos_df.columns:
        train_pos_df = train_pos_df[train_pos_df["label"] == 1][["user_id", "item_id"]]
    else:
        train_pos_df = train_pos_df[["user_id", "item_id"]].copy()

    val_labeled = ensure_binary_labels(val_df)
    # 为验证集采样负样本，使 val_loss 可比
    val_neg = sample_train_negatives(
        val_labeled[val_labeled["label"] == 1][["user_id", "item_id"]],
        n_items, neg_ratio=neg_ratio, seed=base_seed + 9999
    )
    val_pos = val_labeled[val_labeled["label"] == 1][["user_id", "item_id"]].copy()
    val_pos["label"] = 1
    val_labeled = pd.concat([val_pos, val_neg[val_neg["label"] == 0]], ignore_index=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-5
    )
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_val_hitrate = 0.0
    patience_counter = 0
    patience = config["model"]["early_stop_patience"]
    best_state = None

    train_losses, val_losses, lr_history, val_hitrates = [], [], [], []

    # 验证集正样本（用于计算 HitRate）
    val_pos_df = val_labeled[val_labeled["label"] == 1][["user_id", "item_id"]].copy()
    val_users_items = val_pos_df.groupby("user_id")["item_id"].apply(set).to_dict()

    if not resample:
        static_train = sample_train_negatives(
            train_pos_df, n_items, neg_ratio=neg_ratio, seed=base_seed
        )

    epoch_iter = tqdm(range(config["model"]["epochs"]), desc="NCF+Review Training", unit="epoch")
    for epoch in epoch_iter:
        # ----- Training -----
        if resample:
            train_data = sample_train_negatives(
                train_pos_df, n_items, neg_ratio=neg_ratio, seed=base_seed + epoch
            )
        else:
            train_data = static_train

        loader = DataLoader(
            NCFReviewDataset(train_data, user_emb, item_emb),
            batch_size=config["model"]["batch_size"],
            shuffle=True,
        )

        model.train()
        total_loss = 0
        n_batches = 0

        batch_iter = tqdm(loader, desc=f"  Epoch {epoch + 1}", leave=False, unit="batch")
        for users, items, u_rev, i_rev, labels in batch_iter:
            users = users.to(device)
            items = items.to(device)
            u_rev = u_rev.to(device)
            i_rev = i_rev.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(users, items, u_rev, i_rev)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / n_batches

        # ----- Validation -----
        model.eval()
        val_total_loss = 0.0
        val_n = 0
        val_dataset = NCFReviewDataset(val_labeled, user_emb, item_emb)
        val_loader = DataLoader(val_dataset, batch_size=config["model"]["batch_size"])
        with torch.no_grad():
            for users, items, u_rev, i_rev, labels in val_loader:
                users = users.to(device)
                items = items.to(device)
                u_rev = u_rev.to(device)
                i_rev = i_rev.to(device)
                labels = labels.to(device)
                preds = model(users, items, u_rev, i_rev)
                loss = criterion(preds, labels)
                val_total_loss += loss.item() * len(labels)
                val_n += len(labels)
        val_loss = val_total_loss / val_n if val_n > 0 else float("inf")

        # 计算 HitRate@10（ranking 指标，比 BCELoss 更可靠）
        val_hitrate = 0.0
        val_k = 10
        for uid, relevant_items in val_users_items.items():
            train_items = set(
                train_pos_df[train_pos_df["user_id"] == uid]["item_id"].values
            )
            recs = model.recommend(uid, n_items, val_k, exclude=train_items)
            if any(r in relevant_items for r in recs):
                val_hitrate += 1.0
        val_hitrate /= len(val_users_items) if val_users_items else 1.0

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

        # 用 HitRate 做早停（ranking 指标比 pointwise loss 更可靠）
        if val_hitrate > best_val_hitrate:
            best_val_hitrate = val_hitrate
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                current_lr = optimizer.param_groups[0]["lr"]
                if current_lr <= scheduler.min_lrs[0]:
                    tqdm.write(f"    Early stopping at epoch {epoch + 1} (best HitRate@10={best_val_hitrate:.4f})")
                    break

    if best_state:
        model.load_state_dict(best_state)
    plot_training_history(
        train_losses, val_losses, lr_history,
        "outputs/plots/ncf_review_training.png",
        val_hitrates=val_hitrates
    )
    return model