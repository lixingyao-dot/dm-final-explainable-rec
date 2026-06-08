"""Streamlit demo for recommendation explanations."""

import json
import re
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.base_model.itemcf import ItemCF
from src.explain import TemporalExplainer
from src.model_loading import load_bpr_model, load_ncf_model


# ── Metadata helpers ──

def _load_id_maps(data_dir):
    """Load item_map.json and user_map.json, return reverse mappings."""
    data_path = Path(data_dir)
    item_id_to_asin = {}
    user_id_to_name = {}

    item_map_path = data_path / "item_map.json"
    if item_map_path.exists():
        with open(item_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        item_id_to_asin = {int(v): k for k, v in raw.items()}

    user_map_path = data_path / "user_map.json"
    if user_map_path.exists():
        with open(user_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        user_id_to_name = {int(v): k for k, v in raw.items()}

    return item_id_to_asin, user_id_to_name


def _load_item_reviews(data_dir):
    """Load item_reviews.json for item descriptions."""
    path = Path(data_dir) / "item_reviews.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _item_display_name(item_id, item_id_to_asin):
    asin = item_id_to_asin.get(item_id, "未知")
    return f"{asin} (ID:{item_id})"


def _user_display_name(user_id, user_id_to_name):
    name = user_id_to_name.get(user_id, "未知用户")
    return f"{name} (ID:{user_id})"


def _item_image_url(item_id, item_id_to_asin):
    asin = item_id_to_asin.get(item_id)
    if asin:
        return f"https://images-na.ssl-images-amazon.com/images/P/{asin}.01._SCLZZZZZZZ_SX200_.jpg"
    return None


def _item_amazon_url(item_id, item_id_to_asin):
    asin = item_id_to_asin.get(item_id)
    if asin:
        return f"https://www.amazon.com/dp/{asin}"
    return None


def _item_description(item_id, item_reviews, max_chars=150):
    text = item_reviews.get(str(item_id), "")
    if not text:
        return "暂无评论摘要"
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def _latest_file(path: Path, pattern: str):
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_recommender_options(model_dir):
    model_path = Path(model_dir)
    options = []

    base_ckpt = model_path / "ncf_best.pt"
    if base_ckpt.exists():
        options.append(
            {
                "label": "NCF（基础）",
                "checkpoint": str(base_ckpt),
                "family": "ncf",
                "display_order": 0,
            }
        )

    bpr_ckpt = model_path / "ncf_bpr_best.pt"
    if bpr_ckpt.exists():
        options.append(
            {
                "label": "NCF-BPR",
                "checkpoint": str(bpr_ckpt),
                "family": "bpr",
                "display_order": 1,
            }
        )

    temporal_files = []
    for ckpt in model_path.glob("ncf_temporal_l*.pt"):
        match = re.search(r"ncf_temporal_l([0-9.]+)\.pt$", ckpt.name)
        if match:
            temporal_files.append((float(match.group(1)), ckpt))
    for decay, ckpt in sorted(temporal_files, key=lambda x: x[0]):
        options.append(
            {
                "label": f"NCF+Temporal λ={decay:g}",
                "checkpoint": str(ckpt),
                "family": "ncf",
                "decay_lambda": decay,
                "display_order": 10 + decay,
            }
        )

    return options


def load_recommender_from_option(option, device="cpu"):
    family = option["family"]
    checkpoint = option["checkpoint"]
    if family == "bpr":
        model = load_bpr_model(checkpoint, device=device)
    else:
        model = load_ncf_model(checkpoint, device=device)
    return model


def _select_best_variant(metrics: dict, prefix: str):
    candidates = [
        (name, values)
        for name, values in metrics.items()
        if name == prefix or name.startswith(f"{prefix}_") or name.startswith(f"{prefix}+")
    ]
    if not candidates:
        return None, None
    best_name, best_values = max(
        candidates,
        key=lambda kv: (kv[1].get("HitRate@10", 0.0), kv[1].get("NDCG@10", 0.0)),
    )
    return best_name, best_values


def load_model_compare_data(outputs_dir):
    outputs_path = Path(outputs_dir)
    ensemble_file = _latest_file(outputs_path, "exp_ensemble_*.json")
    temporal_file = _latest_file(outputs_path, "exp_temporal_*.json")

    if ensemble_file is None or temporal_file is None:
        raise FileNotFoundError("未找到模型对比所需的实验结果文件。")

    ensemble_metrics = _load_json(ensemble_file)
    temporal_metrics = _load_json(temporal_file)

    rows = []
    for model_name in ["ItemCF", "NCF"]:
        if model_name in ensemble_metrics:
            rows.append(
                {
                    "model": model_name,
                    "HitRate@10": ensemble_metrics[model_name]["HitRate@10"],
                    "NDCG@10": ensemble_metrics[model_name]["NDCG@10"],
                }
            )

    best_ensemble_name, best_ensemble = _select_best_variant(ensemble_metrics, "NCF+ItemCF")
    if best_ensemble_name and best_ensemble:
        rows.append(
            {
                "model": "NCF+ItemCF",
                "HitRate@10": best_ensemble["HitRate@10"],
                "NDCG@10": best_ensemble["NDCG@10"],
                "variant": best_ensemble_name,
            }
        )

    best_temporal_name, best_temporal = _select_best_variant(temporal_metrics, "NCF_temporal")
    if best_temporal_name and best_temporal:
        rows.append(
            {
                "model": "NCF_temporal",
                "HitRate@10": best_temporal["HitRate@10"],
                "NDCG@10": best_temporal["NDCG@10"],
                "variant": best_temporal_name,
            }
        )

    compare_df = pd.DataFrame(rows)
    compare_df = compare_df[["model", "HitRate@10", "NDCG@10"] + (["variant"] if "variant" in compare_df.columns else [])]
    sources = {
        "ensemble": str(ensemble_file),
        "temporal": str(temporal_file),
    }
    return compare_df, sources


def load_temporal_analysis_data(outputs_dir):
    outputs_path = Path(outputs_dir)
    temporal_file = _latest_file(outputs_path, "exp_temporal_*.json")
    if temporal_file is None:
        raise FileNotFoundError("未找到时序分析结果文件。")

    metrics = _load_json(temporal_file)
    rows = []
    for variant, values in metrics.items():
        match = re.search(r"NCF_temporal_λ=([0-9.]+)", variant)
        if not match:
            continue
        decay = float(match.group(1))
        rows.append(
            {
                "lambda": decay,
                "HitRate@10": values.get("HitRate@10", 0.0),
                "NDCG@10": values.get("NDCG@10", 0.0),
                "variant": variant,
            }
        )

    df = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    if df.empty:
        raise ValueError("时序分析结果文件中没有可用的 λ 数据。")

    best_row = df.sort_values(["HitRate@10", "NDCG@10"], ascending=[False, False]).iloc[0]
    meta = {
        "source": str(temporal_file),
        "best_lambda": float(best_row["lambda"]),
        "best_variant": best_row["variant"],
    }
    return df, meta


def load_app_data(data_dir, model_dir, recommender_label=None):
    data_path = Path(data_dir)
    model_path = Path(model_dir)

    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    with open(data_path / "stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    n_users = int(stats["n_users"])
    n_items = int(stats["n_items"])
    itemcf = ItemCF(train_df, n_users=n_users, n_items=n_items)
    recommender_options = discover_recommender_options(model_path)
    if not recommender_options:
        raise FileNotFoundError(f"在 {model_path} 中未找到可用的推荐模型 checkpoint。")

    if recommender_label is None:
        selected_option = next((opt for opt in recommender_options if opt["label"] == "NCF（基础）"), recommender_options[0])
    else:
        selected_option = next((opt for opt in recommender_options if opt["label"] == recommender_label), None)
        if selected_option is None:
            available = ", ".join(opt["label"] for opt in recommender_options)
            raise ValueError(f"未找到模型 {recommender_label}。可用模型: {available}")

    recommender = load_recommender_from_option(selected_option)
    temporal_explainer = TemporalExplainer(train_df, itemcf_model=itemcf)

    # Load metadata for display
    item_id_to_asin, user_id_to_name = _load_id_maps(data_path)
    item_reviews = _load_item_reviews(data_path)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "stats": stats,
        "itemcf": itemcf,
        "recommender": recommender,
        "recommender_label": selected_option["label"],
        "recommender_checkpoint": selected_option["checkpoint"],
        "available_recommenders": recommender_options,
        "temporal_explainer": temporal_explainer,
        "item_id_to_asin": item_id_to_asin,
        "user_id_to_name": user_id_to_name,
        "item_reviews": item_reviews,
        "data_dir": str(data_path),
        "model_dir": str(model_path),
    }


def _build_user_bundle(data, user_id):
    train_df = data["train_df"]
    itemcf = data["itemcf"]
    recommender = data["recommender"]
    temporal_explainer = data["temporal_explainer"]
    n_items = int(data["stats"]["n_items"])
    train_items = set(train_df.loc[train_df["user_id"] == user_id, "item_id"].tolist())
    candidate_items = [i for i in range(n_items) if i not in train_items]
    scores = recommender.score_items(int(user_id), candidate_items)
    order = scores.argsort()[::-1][:10]

    bundle = []
    for idx in order:
        item_id = int(candidate_items[idx])
        score = float(scores[idx])
        itemcf_exp = itemcf.explain(int(user_id), int(item_id), k=5)
        temporal_exp = temporal_explainer.explain(int(user_id), int(item_id), k=5)
        bundle.append(
            {
                "item_id": int(item_id),
                "score": score,
                "itemcf": itemcf_exp,
                "temporal": temporal_exp,
            }
        )
    return bundle


CARD_CSS = """
<style>
.rec-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s;
}
.rec-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}
.rec-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
}
.rec-img {
    width: 100px;
    height: 100px;
    object-fit: cover;
    border-radius: 8px;
    border: 1px solid #eee;
    background: #f8f8f8;
}
.rec-title {
    font-size: 1.1em;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0 0 4px 0;
}
.rec-score {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.85em;
    font-weight: 600;
}
.rec-desc {
    background: #f5f7fa;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.9em;
    color: #444;
    line-height: 1.5;
}
.rec-exp-label {
    font-weight: 600;
    color: #1565c0;
    margin-top: 10px;
    margin-bottom: 4px;
}
.rec-link {
    display: inline-block;
    margin-top: 8px;
    color: #1976d2;
    text-decoration: none;
    font-size: 0.88em;
}
.rec-link:hover { text-decoration: underline; }
</style>
"""


def _render_rec_card(entry, data):
    """Render a single recommendation card using HTML/CSS."""
    import streamlit as st

    item_id = entry["item_id"]
    item_id_to_asin = data["item_id_to_asin"]
    item_reviews = data["item_reviews"]

    display_name = _item_display_name(item_id, item_id_to_asin)
    img_url = _item_image_url(item_id, item_id_to_asin)
    amazon_url = _item_amazon_url(item_id, item_id_to_asin)
    desc = _item_description(item_id, item_reviews, max_chars=200)

    if img_url:
        img_html = (
            f'<img src="{img_url}" class="rec-img" '
            f'onerror="this.onerror=null;this.src=\'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><rect fill=%22%23f0f0f0%22 width=%22100%22 height=%22100%22/><text x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 fill=%22%23999%22 font-size=%2212%22>无图片</text></svg>\';">'
        )
    else:
        img_html = '<div class="rec-img" style="display:flex;align-items:center;justify-content:center;background:#f0f0f0;color:#999;font-size:12px;">无图片</div>'
    link_html = (
        f'<a href="{amazon_url}" target="_blank" class="rec-link">在 Amazon 查看 &rarr;</a>'
        if amazon_url else ""
    )

    html = f"""
    <div class="rec-card">
        <div class="rec-header">
            {img_html}
            <div>
                <p class="rec-title">{display_name}</p>
                <span class="rec-score">推荐分数: {entry['score']:.4f}</span>
            </div>
        </div>
        <div class="rec-desc">{desc}</div>
        <p class="rec-exp-label">协同推理</p>
        <p style="margin:0 0 4px 0; font-size:0.92em; color:#333;">{entry['itemcf']['explanation']}</p>
        <p class="rec-exp-label">时序归因</p>
        <p style="margin:0; font-size:0.92em; color:#333;">{entry['temporal']['explanation']}</p>
        {link_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def main():
    import streamlit as st
    import plotly.express as px

    st.set_page_config(page_title="可解释推荐系统", layout="wide")
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.title("可解释推荐系统")

    recommender_options = discover_recommender_options("outputs/models")
    recommender_labels = [opt["label"] for opt in recommender_options]
    selected_label = st.sidebar.selectbox("推荐模型", recommender_labels, index=0 if recommender_labels else 0)

    data = load_app_data("data/processed", "outputs/models", recommender_label=selected_label)
    st.caption(
        f"推荐模型: `{data['recommender_label']}` | checkpoint: `{data['recommender_checkpoint']}` | "
        f"模型类: `{data['recommender'].__class__.__name__}`"
    )
    test_users = sorted(data["test_df"]["user_id"].unique().tolist())
    if not test_users:
        st.warning("测试集中没有用户。")
        return

    user_id = st.sidebar.selectbox("用户ID", test_users, index=0)
    user_display = _user_display_name(int(user_id), data["user_id_to_name"])
    user_bundle = _build_user_bundle(data, int(user_id))

    tab_rec, tab_model, tab_temporal = st.tabs(["推荐解释", "模型对比", "时序衰减分析"])

    with tab_rec:
        st.subheader(f"用户 {user_display} 的推荐结果")
        for entry in user_bundle[:10]:
            _render_rec_card(entry, data)

    with tab_model:
        st.subheader("模型对比")
        try:
            compare_df, sources = load_model_compare_data("outputs")
            st.caption(f"结果来源: `{sources['ensemble']}` | `{sources['temporal']}`")
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
            fig = px.bar(
                compare_df,
                x="model",
                y=["HitRate@10", "NDCG@10"],
                barmode="group",
                title="HitRate@10 / NDCG@10 对比",
                labels={"value": "指标值", "model": "模型", "variable": "指标"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Highlight best model and ItemCF baseline
            best_row = compare_df.loc[compare_df["HitRate@10"].idxmax()]
            best_model = best_row["model"]
            best_hr = best_row["HitRate@10"]
            itemcf_row = compare_df[compare_df["model"] == "ItemCF"]
            if not itemcf_row.empty:
                itemcf_hr = itemcf_row.iloc[0]["HitRate@10"]
                if itemcf_hr > 0:
                    improvement = (best_hr - itemcf_hr) / itemcf_hr * 100
                    st.success(
                        f"**最佳模型：{best_model}** (HitRate@10={best_hr:.3f})，"
                        f"相比 ItemCF 可解释基线 (HitRate@10={itemcf_hr:.3f}) 提升 {improvement:.1f}%"
                    )
            else:
                st.success(f"**最佳模型：{best_model}** (HitRate@10={best_hr:.3f})")

            if not itemcf_row.empty:
                st.info("**ItemCF** 作为可解释基线，虽然准确率略低，但每条推荐均可提供「因为买过 X、Y 的人也买了这个」的协同推理解释。")
        except Exception as exc:
            st.error(f"模型对比数据加载失败：{exc}")

    with tab_temporal:
        st.subheader("时序衰减分析")
        try:
            curve_df, curve_meta = load_temporal_analysis_data("outputs")
            st.caption(
                f"结果来源: `{curve_meta['source']}` | 最佳 λ = {curve_meta['best_lambda']:.1f}"
            )
            fig = px.line(
                curve_df,
                x="lambda",
                y="HitRate@10",
                markers=True,
                title="λ 与 HitRate@10 的关系",
                labels={"lambda": "λ", "HitRate@10": "HitRate@10"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(curve_df[["lambda", "HitRate@10", "NDCG@10", "variant"]], use_container_width=True, hide_index=True)
            if user_bundle:
                first = user_bundle[0]["temporal"]
                st.markdown(f"**当前推荐项的时序解释：** {first['explanation']}")
                st.json(first)
            else:
                st.info("没有可展示的推荐结果。")
        except Exception as exc:
            st.error(f"时序分析数据加载失败：{exc}")


if __name__ == "__main__":
    main()
