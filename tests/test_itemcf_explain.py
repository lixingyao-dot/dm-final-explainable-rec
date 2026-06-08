import pandas as pd

from src.base_model.itemcf import ItemCF


def test_itemcf_explain_returns_bridge_items_and_text():
    train = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1],
            "item_id": [0, 1, 1, 2],
        }
    )
    model = ItemCF(train, n_users=2, n_items=3, k_neighbors=2)

    result = model.explain(user_id=0, item_id=2, k=2)

    assert result["recommended_item"] == 2
    assert len(result["bridge_items"]) <= 2
    assert result["bridge_items"]
    assert "item_id" in result["bridge_items"][0]
    assert "similarity" in result["bridge_items"][0]
    assert "推荐该商品" in result["explanation"]
