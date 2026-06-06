import json
import tempfile
from pathlib import Path

import pandas as pd
import torch

from experiments.exp6_explain import build_explanations
from src.ncf_models.ncf import NCF


def test_build_explanations_writes_json():
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        tmp_root = Path(tmp_dir)
        data_dir = tmp_root / "processed"
        model_dir = tmp_root / "models"
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

        out = tmp_root / "explanations.json"
        result = build_explanations(
            data_dir=data_dir,
            model_dir=model_dir,
            output_path=out,
            max_users=1,
        )

        assert out.exists()
        assert result["n_users"] == 1
        assert result["output_path"] == str(out)
        assert result["recommender_checkpoint"].endswith("ncf_best.pt")
        assert "recommendations" in result
        assert "score" in result["recommendations"][0]
