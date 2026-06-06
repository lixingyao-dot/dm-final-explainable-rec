"""Helpers for loading trained recommendation checkpoints."""

from pathlib import Path

import torch

from config import CONFIG
from src.ncf_models.ncf import NCF
from src.ncf_models.ncf_bpr import NCFBPR


def _load_state_dict(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _infer_ncf_shapes(state_dict):
    user_emb = state_dict["user_emb_gmf.weight"]
    item_emb = state_dict["item_emb_gmf.weight"]
    n_users, embedding_dim = user_emb.shape
    n_items = item_emb.shape[0]

    mlp_layers = []
    for key in sorted(
        (k for k in state_dict.keys() if k.startswith("mlp.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    ):
        mlp_layers.append(int(state_dict[key].shape[0]))

    if not mlp_layers:
        mlp_layers = list(CONFIG["model"]["ncf_mlp_layers"])

    return n_users, n_items, embedding_dim, tuple(mlp_layers)


def _infer_bpr_shapes(state_dict):
    user_emb = state_dict["user_emb_gmf.weight"]
    item_emb = state_dict["item_emb_gmf.weight"]
    n_users, embedding_dim = user_emb.shape
    n_items = item_emb.shape[0]

    mlp_layers = []
    for key in sorted(
        (k for k in state_dict.keys() if k.startswith("mlp.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    ):
        mlp_layers.append(int(state_dict[key].shape[0]))

    if not mlp_layers:
        mlp_layers = list(CONFIG["model"]["ncf_mlp_layers"])

    return n_users, n_items, embedding_dim, tuple(mlp_layers)


def load_ncf_model(checkpoint_path, device="cpu"):
    """Load a vanilla NCF checkpoint and return a ready-to-use model."""
    checkpoint_path = Path(checkpoint_path)
    state = _load_state_dict(checkpoint_path)
    n_users, n_items, embedding_dim, mlp_layers = _infer_ncf_shapes(state)
    model = NCF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_bpr_model(checkpoint_path, device="cpu"):
    """Load an NCF-BPR checkpoint and return a ready-to-use model."""
    checkpoint_path = Path(checkpoint_path)
    state = _load_state_dict(checkpoint_path)
    n_users, n_items, embedding_dim, mlp_layers = _infer_bpr_shapes(state)
    model = NCFBPR(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        mlp_layers=mlp_layers,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_best_recommender(model_dir="outputs/models", device="cpu"):
    """Load the best available trained recommender checkpoint."""
    model_dir = Path(model_dir)
    candidates = [
        ("ncf_best.pt", load_ncf_model),
        ("ncf_bpr_best.pt", load_bpr_model),
    ]
    for filename, loader in candidates:
        ckpt = model_dir / filename
        if ckpt.exists():
            return loader(ckpt, device=device), ckpt
    raise FileNotFoundError(f"No recommender checkpoint found in {model_dir}")
