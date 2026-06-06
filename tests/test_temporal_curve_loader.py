import json
import tempfile
from pathlib import Path

from app.streamlit_app import load_temporal_analysis_data


def test_load_temporal_analysis_data_builds_curve_table():
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        outputs_dir = Path(tmp_dir) / "outputs"
        outputs_dir.mkdir()
        payload = {
            "NCF_temporal_λ=0.5": {"HitRate@10": 0.3493, "NDCG@10": 0.1894},
            "NCF_temporal_λ=1.5": {"HitRate@10": 0.3682, "NDCG@10": 0.2002},
            "NCF_temporal_λ=5.0": {"HitRate@10": 0.3922, "NDCG@10": 0.2158},
        }
        (outputs_dir / "exp_temporal_20260606_043927.json").write_text(json.dumps(payload), encoding="utf-8")

        df, meta = load_temporal_analysis_data(outputs_dir)

        assert list(df["lambda"]) == [0.5, 1.5, 5.0]
        assert df.loc[df["lambda"] == 5.0, "HitRate@10"].iloc[0] == 0.3922
        assert meta["best_lambda"] == 5.0
        assert meta["source"].endswith("exp_temporal_20260606_043927.json")
