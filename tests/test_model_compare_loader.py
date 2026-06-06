import json
import tempfile
from pathlib import Path

from app.streamlit_app import load_model_compare_data


def test_load_model_compare_data_builds_rows_from_outputs():
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        outputs_dir = Path(tmp_dir) / "outputs"
        outputs_dir.mkdir()

        ensemble = {
            "ItemCF": {"HitRate@10": 0.339, "NDCG@10": 0.1969},
            "NCF": {"HitRate@10": 0.3412, "NDCG@10": 0.1828},
            "NCF+ItemCF_a=0.3": {"HitRate@10": 0.3637, "NDCG@10": 0.2046},
            "NCF+ItemCF_a=0.4": {"HitRate@10": 0.3610, "NDCG@10": 0.2031},
        }
        temporal = {
            "NCF_temporal_λ=1.0": {"HitRate@10": 0.3539, "NDCG@10": 0.1920},
            "NCF_temporal_λ=1.5": {"HitRate@10": 0.3682, "NDCG@10": 0.2002},
            "NCF_temporal_λ=2.0": {"HitRate@10": 0.3627, "NDCG@10": 0.1985},
        }

        (outputs_dir / "exp_ensemble_20260606_045157.json").write_text(
            json.dumps(ensemble), encoding="utf-8"
        )
        (outputs_dir / "exp_temporal_20260606_043927.json").write_text(
            json.dumps(temporal), encoding="utf-8"
        )

        df, sources = load_model_compare_data(outputs_dir)

        assert list(df["model"]) == ["ItemCF", "NCF", "NCF+ItemCF", "NCF_temporal"]
        assert df.loc[df["model"] == "NCF+ItemCF", "HitRate@10"].iloc[0] == 0.3637
        assert df.loc[df["model"] == "NCF_temporal", "HitRate@10"].iloc[0] == 0.3682
        assert sources["ensemble"].endswith("exp_ensemble_20260606_045157.json")
        assert sources["temporal"].endswith("exp_temporal_20260606_043927.json")
