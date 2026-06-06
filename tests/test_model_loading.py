import tempfile
from pathlib import Path

import torch

from src.model_loading import load_ncf_model
from src.ncf_models.ncf import NCF


def test_load_ncf_model_loads_checkpoint_and_scores():
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        root = Path(tmp_dir)
        ckpt = root / "ncf_best.pt"

        model = NCF(n_users=3, n_items=4, embedding_dim=2, mlp_layers=(4, 2, 1))
        torch.save(model.state_dict(), ckpt)

        loaded = load_ncf_model(ckpt)

        assert loaded.n_users == 3
        assert loaded.n_items == 4
        scores = loaded.score_items(0, [0, 1, 2, 3])
        assert len(scores) == 4
