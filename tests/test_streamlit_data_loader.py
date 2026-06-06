import json
import tempfile
from pathlib import Path

import pandas as pd
import torch

from app.streamlit_app import load_app_data
from src.ncf_models.ncf import NCF


def test_load_app_data_returns_expected_keys():
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        root = Path(tmp_dir)
        data_dir = root / "processed"
        model_dir = root / "models"
        data_dir.mkdir()
        model_dir.mkdir()

        train = pd.DataFrame(
            {
                "user_id": [0, 0, 1],
                "item_id": [0, 1, 1],
                "timestamp": [100, 200, 150],
            }
        )
        test = pd.DataFrame(
            {
                "user_id": [0],
                "item_id": [2],
                "timestamp": [300],
            }
        )
        train.to_csv(data_dir / "train.csv", index=False)
        test.to_csv(data_dir / "test.csv", index=False)
        with open(data_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump({"n_users": 2, "n_items": 3}, f)

        model = NCF(n_users=2, n_items=3, embedding_dim=2, mlp_layers=(4, 2, 1))
        torch.save(model.state_dict(), model_dir / "ncf_best.pt")

        data = load_app_data(data_dir, model_dir)

        assert "train_df" in data
        assert "test_df" in data
        assert "stats" in data
        assert "recommender" in data
        assert "temporal_explainer" in data
        assert data["recommender_checkpoint"].endswith("ncf_best.pt")
