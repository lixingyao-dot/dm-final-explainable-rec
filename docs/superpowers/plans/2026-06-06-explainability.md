# Explainability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add item-based and temporal explanations to the existing recommender, generate batch explanation outputs, and provide a lightweight Streamlit demo that reads the shipped model and processed dataset.

**Architecture:** Extend the existing `ItemCF` model with an `explain()` method that reuses its similarity matrix and user history. Add a small temporal explainer in `src/explain.py` that reads a user's history and orders it by the same recency weighting logic already used in training. Then add a batch script that loads processed data and saved model artifacts, writes explanation JSON, and expose a read-only Streamlit app through small helper functions so the UI is easy to smoke test without browser automation.

**Tech Stack:** Python, pandas, NumPy, SciPy, PyTorch, Streamlit, Plotly, pytest.

---

### Task 1: Add `ItemCF.explain()`

**Files:**
- Modify: `src/base_model/itemcf.py`
- Test: `tests/test_itemcf_explain.py`

- [ ] **Step 1: Write the failing test**

```python
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
    assert "explanation" in result
    assert result["bridge_items"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m pytest tests/test_itemcf_explain.py -v`
Expected: FAIL because `ItemCF` has no `explain()` method yet.

- [ ] **Step 3: Write minimal implementation**

```python
def explain(self, user_id, item_id, k=5):
    if user_id >= self.n_users or item_id >= self.n_items:
        return {"recommended_item": int(item_id), "bridge_items": [], "explanation": "Not enough history to explain."}
    user_row = self.user_item[user_id].toarray().ravel()
    interacted = np.where(user_row > 0)[0]
    if len(interacted) == 0:
        return {"recommended_item": int(item_id), "bridge_items": [], "explanation": "No user history found."}
    item_sims = self.similarities[item_id].toarray().ravel()
    bridges = []
    for hist_item in interacted:
        if hist_item == item_id:
            continue
        bridges.append({"item_id": int(hist_item), "similarity": float(item_sims[hist_item])})
    bridges.sort(key=lambda x: x["similarity"], reverse=True)
    bridges = bridges[:k]
    bridge_ids = [b["item_id"] for b in bridges]
    if bridge_ids:
        explanation = f"Recommended because you interacted with {', '.join(map(str, bridge_ids))}, which are similar to {item_id}."
    else:
        explanation = f"Recommended because item {item_id} is similar to your history."
    return {"recommended_item": int(item_id), "bridge_items": bridges, "explanation": explanation}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m pytest tests/test_itemcf_explain.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/base_model/itemcf.py tests/test_itemcf_explain.py
git commit -m "feat: add itemcf explanations"
```

### Task 2: Add temporal explanation helper

**Files:**
- Modify: `src/explain.py`
- Test: `tests/test_temporal_explainer.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from src.explain import TemporalExplainer


def test_temporal_explainer_sorts_by_recency_weight():
    train = pd.DataFrame(
        {
            "user_id": [0, 0, 0],
            "item_id": [10, 11, 12],
            "timestamp": [100, 200, 300],
        }
    )
    explainer = TemporalExplainer(train)
    result = explainer.explain(user_id=0, item_id=99, k=2)

    assert result["top_weighted_items"][0]["item_id"] == 12
    assert len(result["top_weighted_items"]) == 2
    assert "explanation" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m pytest tests/test_temporal_explainer.py -v`
Expected: FAIL because `TemporalExplainer` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class TemporalExplainer:
    def __init__(self, train_df, decay_lambda=2.0):
        self.train_df = train_df.copy()
        self.decay_lambda = decay_lambda
        if "time_weight" not in self.train_df.columns:
            from src.temporal import compute_time_weights
            self.train_df = compute_time_weights(self.train_df, decay_lambda=decay_lambda)

    def explain(self, user_id, item_id, k=5):
        user_hist = self.train_df[self.train_df["user_id"] == user_id].copy()
        if user_hist.empty:
            return {"top_weighted_items": [], "explanation": "No historical interactions found."}
        user_hist = user_hist.sort_values(["time_weight", "timestamp"], ascending=[False, False]).head(k)
        items = [
            {
                "item_id": int(row.item_id),
                "days_ago": None,
                "weight": round(float(row.time_weight), 4),
            }
            for row in user_hist.itertuples(index=False)
        ]
        explanation = "Recommended mainly from recent behavior: " + ", ".join(f"{x['item_id']} ({x['weight']})" for x in items)
        return {"top_weighted_items": items, "explanation": explanation}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m pytest tests/test_temporal_explainer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/explain.py tests/test_temporal_explainer.py
git commit -m "feat: add temporal explanations"
```

### Task 3: Add batch explanation generation

**Files:**
- Create: `experiments/exp6_explain.py`
- Test: `tests/test_exp6_explain.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from experiments.exp6_explain import build_explanations


def test_build_explanations_writes_json(tmp_path):
    out = tmp_path / "explanations.json"
    result = build_explanations(
        data_dir=Path("data/processed"),
        model_dir=Path("outputs/models"),
        output_path=out,
        max_users=2,
    )
    assert out.exists()
    assert result["n_users"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m pytest tests/test_exp6_explain.py -v`
Expected: FAIL because the module and function do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def build_explanations(data_dir, model_dir, output_path, max_users=None):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m pytest tests/test_exp6_explain.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp6_explain.py tests/test_exp6_explain.py
git commit -m "feat: add batch explanation export"
```

### Task 4: Add a lightweight Streamlit demo

**Files:**
- Create: `app/streamlit_app.py`
- Test: `tests/test_streamlit_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
from app.streamlit_app import load_app_data


def test_load_app_data_returns_expected_keys():
    data = load_app_data("data/processed", "outputs/models")
    assert "train_df" in data
    assert "stats" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m pytest tests/test_streamlit_data_loader.py -v`
Expected: FAIL because the module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def load_app_data(data_dir, model_dir):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m pytest tests/test_streamlit_data_loader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py tests/test_streamlit_data_loader.py
git commit -m "feat: add explainability demo app"
```

### Task 5: End-to-end verification

**Files:**
- No new files; verify the full flow.

- [ ] **Step 1: Run the focused test suite**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m pytest tests/test_itemcf_explain.py tests/test_temporal_explainer.py tests/test_exp6_explain.py tests/test_streamlit_data_loader.py -v`
Expected: PASS.

- [ ] **Step 2: Run a smoke execution of the batch export**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe experiments/exp6_explain.py`
Expected: writes `outputs/exp6_explanations.json` without errors.

- [ ] **Step 3: Launch the app manually**

Run: `C:\DownloadApp\miniconda3\envs\yolov11\python.exe -m streamlit run app/streamlit_app.py`
Expected: app starts and loads `data/processed` plus `outputs/models`.

- [ ] **Step 4: Inspect generated output**

Confirm the JSON contains at least one user record with `itemcf_explanation` and `temporal_explanation`.

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: complete explainability workflow"
```
