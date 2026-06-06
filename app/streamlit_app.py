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


def main():
    import streamlit as st
    import plotly.express as px

    st.set_page_config(page_title="可解释推荐系统", layout="wide")
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
    user_bundle = _build_user_bundle(data, int(user_id))

    tab_rec, tab_model, tab_temporal = st.tabs(["推荐解释", "模型对比", "时序分析"])

    with tab_rec:
        st.subheader(f"用户 {user_id} 的推荐结果")
        for entry in user_bundle[:5]:
            with st.expander(f"物品 {entry['item_id']}  分数={entry['score']:.4f}"):
                st.markdown(f"**ItemCF：** {entry['itemcf']['explanation']}")
                st.markdown(f"**Temporal：** {entry['temporal']['explanation']}")

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
        except Exception as exc:
            st.error(f"模型对比数据加载失败：{exc}")

    with tab_temporal:
        st.subheader("时序解释")
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
