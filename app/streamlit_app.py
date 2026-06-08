"""Streamlit demo for recommendation explanations."""

import json
import re
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.base_model.itemcf import ItemCF
from src.base_model.popularity import PopularityRecommender
from src.base_model.ta_itemcf import TAItemCF
from src.base_model.usercf import UserCF
from src.explain import TemporalExplainer
from src.model_loading import load_bpr_model, load_ncf_model


class BaseRecommenderWrapper:
    """Wraps base models (no score_items) to match NCF's score_items interface."""

    def __init__(self, model, n_items):
        self._model = model
        self._n_items = n_items

    def score_items(self, user_id, items):
        recs = self._model.recommend(int(user_id), self._n_items, k=len(items))
        n = len(recs)
        rank_map = {item: (n - idx) / n for idx, item in enumerate(recs)}
        return np.array([rank_map.get(i, 0.0) for i in items])

    def __getattr__(self, name):
        return getattr(self._model, name)


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
            }
        )

    temporal_ckpt = model_path / "ncf_temporal_l5.0.pt"
    if temporal_ckpt.exists():
        options.append(
            {
                "label": "NCF+Temporal λ=5",
                "checkpoint": str(temporal_ckpt),
                "family": "ncf",
                "decay_lambda": 5.0,
            }
        )

    for base_type, label in [
        ("popularity", "Popularity"),
        ("itemcf", "ItemCF"),
        ("usercf", "UserCF"),
        ("ta_itemcf", "TAItemCF"),
    ]:
        options.append({"label": label, "family": "base", "base_type": base_type})

    return options


def load_recommender_from_option(option, train_df=None, n_users=None, n_items=None, device="cpu"):
    family = option["family"]
    if family == "base":
        base_type = option["base_type"]
        if base_type == "popularity":
            model = PopularityRecommender(train_df)
        elif base_type == "itemcf":
            model = ItemCF(train_df, n_users=n_users, n_items=n_items)
        elif base_type == "usercf":
            model = UserCF(train_df, n_users=n_users, n_items=n_items)
        elif base_type == "ta_itemcf":
            model = TAItemCF(train_df, n_users=n_users, n_items=n_items)
        else:
            raise ValueError(f"未知的 base 模型类型: {base_type}")
        return BaseRecommenderWrapper(model, n_items)
    elif family == "bpr":
        return load_bpr_model(option["checkpoint"], device=device)
    else:
        return load_ncf_model(option["checkpoint"], device=device)


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

    recommender = load_recommender_from_option(
        selected_option, train_df=train_df, n_users=n_users, n_items=n_items
    )
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
        "recommender_checkpoint": selected_option.get("checkpoint", "N/A"),
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
    raw_scores = recommender.score_items(int(user_id), candidate_items)

    # Base models use synthetic rank-based scores (no real raw score)
    has_raw = not isinstance(recommender, BaseRecommenderWrapper)

    # Normalize scores to [0, 1] so different models are comparable
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        norm_scores = (raw_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.ones_like(raw_scores)
    order = norm_scores.argsort()[::-1][:10]

    bundle = []
    for idx in order:
        item_id = int(candidate_items[idx])
        itemcf_exp = itemcf.explain(int(user_id), int(item_id), k=5)
        temporal_exp = temporal_explainer.explain(int(user_id), int(item_id), k=5)
        bundle.append(
            {
                "item_id": int(item_id),
                "score": float(norm_scores[idx]),
                "raw_score": float(raw_scores[idx]) if has_raw else None,
                "itemcf": itemcf_exp,
                "temporal": temporal_exp,
            }
        )
    return bundle


CARD_CSS = """
<style>
/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 32px 40px;
    border-radius: 16px;
    margin-bottom: 28px;
    box-shadow: 0 4px 20px rgba(102,126,234,0.3);
}
.hero-banner h2 { margin: 0 0 6px 0; font-size: 1.6em; font-weight: 700; }
.hero-banner p  { margin: 0; font-size: 1.05em; opacity: 0.9; }

/* ── Product Grid ── */
.product-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
}
@media (max-width: 800px) { .product-grid { grid-template-columns: 1fr; } }

/* ── Product Card ── */
.product-card {
    background: #ffffff;
    border: 1px solid #e8e8e8;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
    display: flex;
    flex-direction: column;
}
.product-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.product-img-wrap {
    width: 100%;
    height: 180px;
    overflow: hidden;
    background: #f5f5f5;
    display: flex;
    align-items: center;
    justify-content: center;
}
.product-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.product-body {
    padding: 16px 18px;
    flex: 1;
    display: flex;
    flex-direction: column;
}
.product-title {
    font-size: 1.05em;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0 0 8px 0;
    line-height: 1.4;
}
.score-badge {
    display: inline-block;
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    color: #0a5c36;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.82em;
    font-weight: 700;
    margin-bottom: 10px;
    width: fit-content;
}
.product-desc {
    background: #f8f9fb;
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.88em;
    color: #555;
    line-height: 1.55;
    flex: 1;
}
.product-link {
    display: inline-block;
    margin-top: 10px;
    color: #667eea;
    text-decoration: none;
    font-size: 0.85em;
    font-weight: 600;
}
.product-link:hover { text-decoration: underline; }

/* ── Explanation labels ── */
.rec-exp-label {
    font-weight: 600;
    color: #667eea;
    margin-top: 10px;
    margin-bottom: 4px;
    font-size: 0.9em;
}
</style>
"""


def _render_ecommerce_card(entry, data):
    """Render a product card for the e-commerce homepage."""
    import streamlit as st

    item_id = entry["item_id"]
    item_id_to_asin = data["item_id_to_asin"]
    item_reviews = data["item_reviews"]

    display_name = _item_display_name(item_id, item_id_to_asin)
    img_url = _item_image_url(item_id, item_id_to_asin)
    amazon_url = _item_amazon_url(item_id, item_id_to_asin)
    desc = _item_description(item_id, item_reviews, max_chars=120)

    if img_url:
        img_tag = (
            f'<img src="{img_url}" '
            f'onerror="this.onerror=null;this.src=\'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22300%22 height=%22180%22><rect fill=%22%23f0f0f0%22 width=%22300%22 height=%22180%22/><text x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 fill=%22%23aaa%22 font-size=%2214%22>暂无图片</text></svg>\';">'
        )
    else:
        img_tag = '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#aaa;font-size:14px;">暂无图片</div>'

    link_html = (
        f'<a href="{amazon_url}" target="_blank" class="product-link">在 Amazon 查看 &rarr;</a>'
        if amazon_url else ""
    )

    raw = entry.get("raw_score")
    raw_hint = f'<span title="模型原始分: {raw:.4f}" style="cursor:help;font-size:0.78em;color:#888;margin-left:6px;">ⓘ</span>' if raw is not None else ""

    html = f"""
    <div class="product-card">
        <div class="product-img-wrap">{img_tag}</div>
        <div class="product-body">
            <p class="product-title">{display_name}</p>
            <div style="margin-bottom:10px;">
                <span class="score-badge">推荐度 {entry['score']:.4f}</span>{raw_hint}
            </div>
            <div class="product-desc">{desc}</div>
            {link_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

    with st.expander("为什么推荐这个？"):
        st.markdown(f"**协同推理：** {entry['itemcf']['explanation']}")
        st.markdown(f"**时序归因：** {entry['temporal']['explanation']}")


def _render_home_page(data, user_id, user_bundle):
    """Render the e-commerce style recommendation homepage."""
    import streamlit as st

    user_display = _user_display_name(int(user_id), data["user_id_to_name"])

    st.markdown(
        f"""<div class="hero-banner">
            <h2>你好，{user_display}</h2>
            <p>为你精选了 {len(user_bundle)} 件你可能喜欢的商品</p>
        </div>""",
        unsafe_allow_html=True,
    )

    for row_start in range(0, len(user_bundle), 2):
        cols = st.columns(2)
        for col, entry in zip(cols, user_bundle[row_start : row_start + 2]):
            with col:
                _render_ecommerce_card(entry, data)


def _render_analysis_page(data, user_bundle):
    """Render the system analysis page (model comparison + temporal analysis)."""
    import streamlit as st
    import plotly.express as px

    st.header("系统分析")

    tab_model, tab_temporal = st.tabs(["模型对比", "时序衰减分析"])

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


def main():
    import streamlit as st

    st.set_page_config(page_title="可解释推荐系统", layout="wide")
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    # ── Sidebar: navigation + controls ──
    page = st.sidebar.radio("页面", ["推荐首页", "系统分析"], horizontal=True)

    recommender_options = discover_recommender_options("outputs/models")
    recommender_labels = [opt["label"] for opt in recommender_options]
    selected_label = st.sidebar.selectbox("推荐模型", recommender_labels, index=0 if recommender_labels else 0)

    with st.spinner("正在加载数据和模型，请稍候..."):
        data = load_app_data("data/processed", "outputs/models", recommender_label=selected_label)

    test_users = sorted(data["test_df"]["user_id"].unique().tolist())
    if not test_users:
        st.warning("测试集中没有用户。")
        return

    user_id = st.sidebar.selectbox("用户ID", test_users, index=0)

    # ── Sidebar: model info (collapsed) ──
    recommender = data["recommender"]
    model_cls = recommender._model.__class__.__name__ if isinstance(recommender, BaseRecommenderWrapper) else recommender.__class__.__name__
    ckpt = data["recommender_checkpoint"]
    ckpt_display = Path(ckpt).name if ckpt != "N/A" else "无（实时构建）"
    with st.sidebar.expander("模型信息"):
        st.caption(f"模型类: `{model_cls}`")
        st.caption(f"Checkpoint: `{ckpt_display}`")

    # ── Generate recommendations ──
    with st.spinner("正在生成推荐结果..."):
        user_bundle = _build_user_bundle(data, int(user_id))

    # ── Route to selected page ──
    if page == "推荐首页":
        _render_home_page(data, user_id, user_bundle)
    else:
        _render_analysis_page(data, user_bundle)


if __name__ == "__main__":
    main()
