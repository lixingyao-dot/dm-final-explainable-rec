import tempfile
from pathlib import Path

import pandas as pd
import torch

from app.streamlit_app import discover_recommender_options, load_app_data
from src.ncf_models.ncf import NCF
from src.ncf_models.ncf_bpr import NCFBPR


def test_discover_recommender_options_lists_supported_checkpoints():
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        model_dir = Path(tmp_dir) / "models"
        model_dir.mkdir()
        torch.save(NCF(n_users=2, n_items=3, embedding_dim=2, mlp_layers=(4, 2, 1)).state_dict(), model_dir / "ncf_best.pt")
        torch.save(NCFBPR(n_users=2, n_items=3, embedding_dim=2, mlp_layers=(4, 2, 1)).state_dict(), model_dir / "ncf_bpr_best.pt")
        torch.save(NCF(n_users=2, n_items=3, embedding_dim=2, mlp_layers=(4, 2, 1)).state_dict(), model_dir / "ncf_temporal_l1.5.pt")

        options = discover_recommender_options(model_dir)

        labels = [opt["label"] for opt in options]
        assert "NCF（基础）" in labels
        assert "NCF-BPR" in labels
        assert any(label.startswith("NCF+Temporal λ=1.5") for label in labels)


def test_load_app_data_loads_selected_recommender():
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        root = Path(tmp_dir)
        data_dir = root / "processed"
        model_dir = root / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        train = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [0, 1, 1], "timestamp": [100, 200, 150]})
        test = pd.DataFrame({"user_id": [0], "item_id": [2], "timestamp": [300]})
        train.to_csv(data_dir / "train.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)
        (data_dir / "stats.json").write_text('{"n_users": 2, "n_items": 3}', encoding="utf-8")

        torch.save(NCF(n_users=2, n_items=3, embedding_dim=2, mlp_layers=(4, 2, 1)).state_dict(), model_dir / "ncf_best.pt")
        torch.save(NCFBPR(n_users=2, n_items=3, embedding_dim=2, mlp_layers=(4, 2, 1)).state_dict(), model_dir / "ncf_bpr_best.pt")

        data = load_app_data(data_dir, model_dir, recommender_label="NCF-BPR")

        assert data["recommender"].__class__.__name__ == "NCFBPR"
        assert data["recommender_label"] == "NCF-BPR"
