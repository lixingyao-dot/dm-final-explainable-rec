import pandas as pd

from src.base_model.itemcf import ItemCF
from src.explain import TemporalExplainer


def test_temporal_explainer_changes_with_reference_item():
    train = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [0, 1, 0, 2, 1, 3],
            "timestamp": [100, 200, 110, 120, 130, 140],
        }
    )
    itemcf = ItemCF(train, n_users=3, n_items=4)
    explainer = TemporalExplainer(train, itemcf_model=itemcf)

    result_a = explainer.explain(user_id=0, item_id=2, k=2)
    result_b = explainer.explain(user_id=0, item_id=3, k=2)

    assert len(result_a["top_weighted_items"]) == 2
    assert len(result_b["top_weighted_items"]) == 2
    assert result_a["top_weighted_items"][0]["item_id"] != result_b["top_weighted_items"][0]["item_id"]
    assert "for item 2" in result_a["explanation"].lower()
    assert "for item 3" in result_b["explanation"].lower()
