# 基于用户行为与评论文本的可解释推荐系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Explainable Recommendation System** — 一个融合**协同过滤**、**深度学习**、**评论文本语义**与**时间衰减**的可解释推荐系统。系统从传统基线到图神经网络，提供从"推荐了什么"到"为什么推荐"的完整链路。

---

## 目录

- [项目背景与目标](#项目背景与目标)
- [核心功能](#核心功能)
- [技术架构](#技术架构)
  - [模型体系](#模型体系)
  - [研究问题](#研究问题)
  - [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
  - [数据预处理](#数据预处理)
- [一键运行所有实验](#一键运行所有实验)
  - [实验列表](#实验列表)
  - [扩展实验](#扩展实验)
  - [产出文件](#产出文件)
- [运行可视化应用](#运行可视化应用)
- [配置说明](#配置说明)
- [实验结果与解读](#实验结果与解读)
- [常见问题排查](#常见问题排查)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [参考文献](#参考文献)

---

## 项目背景与目标

推荐系统是电商、内容平台的核心组件。传统方法（如协同过滤）仅依赖用户-物品交互矩阵，存在两个关键局限：

1. **冷启动问题** — 新用户/新物品缺乏交互数据，无法做出有效推荐
2. **可解释性缺失** — 只给出推荐结果，无法向用户解释"为什么推荐这个"

本项目针对上述问题，构建一个从**传统基线**到**深度学习**到**可解释分析**的完整推荐系统，系统性地回答以下研究问题：

| 研究问题 | 核心关注 |
|---------|---------|
| **RQ1** | 深度学习模型（NCF）相比传统协同过滤（UserCF / ItemCF）是否有显著提升？ |
| **RQ2** | 融入评论文本语义信息（NCF+Review）能否进一步提升推荐质量？ |
| **RQ3** | 图神经网络（LightGCN）能否更好地建模用户-物品高阶关系？ |
| **RQ4** | 时间衰减加权（Temporal Weighting）能否提升近期行为预测的准确性？ |
| **RQ5** | 多种推荐结果的融合是否能实现更好的效果？ |
| **RQ6** | 推荐结果能否提供可解释的理由？ |

### 数据集

使用 **Amazon Electronics 5-core** 公开数据集（[来源](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)），包含约 33 万条用户评论，覆盖丰富的用户-物品交互和评论文本。

---

## 核心功能

| 功能 | 说明 |
|------|------|
| **多模型推荐** | 支持 6 种推荐算法：Popularity、UserCF、ItemCF、NCF、NCF+Review、LightGCN |
| **时间感知推荐** | 基于时间衰减加权的训练策略，提高近期行为的预测权重 |
| **集成推荐** | NCF + ItemCF 分数级融合（Score-level Ensemble） |
| **可解释推荐** | 关键词重叠分析、相似用户分析、SHAP 值归因三种解释方案 |
| **交互式可视化** | Streamlit 应用，支持模型对比、用户推荐、解释查看、Embedding 可视化 |

---

## 技术架构

### 模型体系

系统设计为**三层递进式**模型体系：

```
Layer 1: 传统基线
  ├── Popularity         全局热门推荐（最低基线）
  ├── UserCF             基于用户的协同过滤
  └── ItemCF             基于物品的协同过滤

Layer 2: 深度模型
  ├── NCF                神经协同过滤（GMF + MLP 双路径融合）
  ├── NCF+BPR            贝叶斯个性化排序（Pairwise Ranking）
  └── MatrixFactorization 经典矩阵分解（SVD）

Layer 3: 创新模型
  ├── NCF+Review         NCF + SBERT 评论文本语义融合
  ├── LightGCN           图卷积网络（User-Item 二部图传播）
  ├── Temporal-NCF       时间衰减加权训练
  ├── NCF+ItemCF Ensemble 分数级集成融合
  └── TA-ItemCF          时间感知物品协同过滤
```

#### 模型细节

**NCF (Neural Collaborative Filtering)** — 核心深度模型

```
用户ID ──→ User Embedding ──→ ┐
                               ├─→ GMF (element-wise product) ─→ ┐
物品ID ──→ Item Embedding ──→ ┘                                  │
                                                                  ├─→ Concat → MLP → Score
用户ID ──→ User Embedding ──→ ┐                                  │
                               ├─→ MLP (多层感知机) ──────────────→ ┘
物品ID ──→ Item Embedding ──→ ┘
```

**NCF+Review** — 语义融合模型

```
用户评论 ──→ SBERT ──→ 384d Review Embedding ──→ Linear Projection ─→ ┐
                                                                        ├─→ α·review + (1-α)·behavior → MLP → Score
NCF行为向量 ───────────────────────────────────────────────────────────→ ┘
```

**LightGCN** — 图神经网络

```
User-Item二部图 → Light Graph Convolution (3层) → 加权平均各层Embedding → BPR Loss
```

### 项目结构

```
dm-final-explainable-rec/
├── app/                        # Streamlit 可视化应用
│   └── streamlit_app.py        # 交互式推荐演示界面
├── src/                        # 核心源代码
│   ├── base_model/             # 传统推荐模型
│   │   ├── popularity.py       # 热门推荐
│   │   ├── usercf.py           # User-based CF
│   │   ├── itemcf.py           # Item-based CF
│   │   └── ta_itemcf.py        # 时间感知 ItemCF
│   ├── ncf_models/             # 神经协同过滤模型族
│   │   ├── ncf.py              # 标准 NCF
│   │   ├── ncf_bpr.py          # NCF + BPR Loss
│   │   └── ncf_review.py       # NCF + 评论文本融合
│   ├── preprocess.py           # 数据预处理管道
│   ├── evaluate.py             # 评估指标 (Recall/NDCG/HR/MAP/Precision)
│   ├── explain.py              # 可解释性模块
│   ├── ensemble.py             # 模型集成
│   ├── temporal.py             # 时间衰减训练
│   ├── lightgcn.py             # LightGCN 模型
│   ├── mf.py                   # 矩阵分解
│   ├── model_loading.py        # 模型加载工具
│   ├── plotting.py             # 训练可视化
│   └── utils.py                # 通用工具函数
├── experiments/                # 实验脚本
│   ├── exp0_hyper_search.py    # 超参数搜索
│   ├── exp1_baseline.py        # 基线对比实验
│   ├── exp1b_ncf_loss.py       # BPR Loss 对比
│   ├── exp2_semantic.py        # 语义贡献分析
│   ├── exp3_graph.py           # 图模型对比
│   ├── exp4_temporal.py        # 时间衰减实验
│   ├── exp5_ensemble.py        # 集成模型实验
│   ├── exp6_explain.py         # 可解释性导出
│   └── exp6_ta_itemcf.py       # TA-ItemCF 扫描
├── tests/                      # 单元测试
├── docs/                       # 文档、开发记录、计划
├── config.py                   # 全局配置
├── run_all.py                  # 一键运行脚本
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

---

## 快速开始

### 环境要求

- **Python**: 3.10+
- **CUDA** (可选): 用于 GPU 加速模型训练
- **磁盘空间**: 至少 10GB（原始数据 + 预处理 + 模型权重）

### 安装步骤

```powershell
# 1. 创建 Python 3.10 环境
conda create -n torch python=3.10 -y
conda activate torch

# 2. 下载数据集
wget -P data/raw https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz

# 3. 安装依赖
pip install -r requirements.txt

# 如果遇到 GPU 兼容性问题，尝试：
pip install -r requirements-gpu-compat.txt
```

### 数据预处理

```powershell
python src/preprocess.py
```

预处理管道包括以下步骤：

| 阶段 | 说明 |
|------|------|
| Stage 1 | 加载原始 JSON 数据，字段审计 |
| Stage 2 | 数据清洗：过滤无效评分、空评论文本 |
| Stage 3 | k-core 过滤：保证每个用户/物品至少有 k=5 次交互 |
| Stage 4 | 活跃用户采样、Train/Val/Test 划分 (70/15/15) |
| Stage 5 | 负采样：每条正样本配 4 个负样本 |
| Stage 6 | SBERT 语义嵌入：将评论文本编码为 384 维向量 |
| Stage 7 | 输出处理后的 CSV 文件和统计数据 |

预处理完成后，`data/processed/` 目录下将生成：

```
data/processed/
├── train.csv              # 训练集
├── val.csv                # 验证集
├── test.csv               # 测试集
├── stats.json             # 数据集统计信息
├── item_map.json          # ASIN → 内部ID 映射
├── user_map.json          # 用户ID → 内部ID 映射
├── user_review_emb.npy    # 用户评论嵌入
├── item_review_emb.npy    # 物品评论嵌入
└── item_reviews.json      # 物品评论聚合文本
```

---

## 一键运行所有实验

```powershell
# 按顺序运行全部实验 (0→1→2→3→4→5→6)
python run_all.py

# 查看实验列表
python run_all.py --list

# 只运行指定实验
python run_all.py --exp 1      # 基线对比
python run_all.py --exp 2      # 语义贡献分析
python run_all.py --exp 3      # 图模型对比
```

### 实验列表

| 编号 | 实验名称 | 对比方案 | 核心指标 | 预期运行时间 |
|------|---------|---------|---------|------------|
| 0 | NCF 超参数搜索 | 搜索最优 embedding_dim × learning_rate | Validation Loss | ~20 分钟 |
| 1 | 基线模型对比 | Popularity / UserCF / ItemCF / NCF | Recall@K, NDCG@K | ~30 分钟 |
| 2 | 语义贡献分析 | NCF vs NCF+Review | Recall@K, NDCG@K | ~40 分钟 |
| 3 | 图模型对比 | NCF vs LightGCN | Recall@K, NDCG@K | ~60 分钟 |
| 4 | 时间衰减实验 | NCF vs NCF+Temporal | Recall@K, NDCG@K | ~30 分钟 |
| 5 | 集成模型 | NCF + ItemCF 集成 | Recall@K, NDCG@K | ~10 分钟 |
| 6 | 可解释性导出 | ItemCF & Temporal 解释 | 解释样例输出 | ~5 分钟 |

> 运行时间基于 CPU (i7) 估算，GPU 可加速 3-5 倍。

### 扩展实验

```powershell
# NCF vs NCF+BPR 对比（训练 + 评估）
python experiments/exp1b_ncf_loss.py --models all --train

# Temporal-Aware ItemCF lambda 参数扫描
python experiments/exp6_ta_itemcf.py --lambdas 0.5 1.0 2.0 3.0 5.0
```

### 产出文件

```
outputs/
├── ncf_hyper/                    # 超参数搜索结果
│   └── results.json
├── models/                       # 训练好的模型权重
│   ├── ncf_best.pt
│   ├── ncf_review_best.pt
│   ├── lightgcn_best.pt
│   ├── ncf_bpr_best.pt
│   └── ncf_temporal_best.pt
└── explanations/                 # 可解释性结果
    ├── itemcf_explanations.json
    └── temporal_explanations.json
```

---

## 运行可视化应用

```powershell
streamlit run app/streamlit_app.py
```

应用提供四个核心页面：

### 1. 模型对比
在测试集上对比各模型的评估指标，支持 Recall@K、NDCG@K、Precision@K、HitRate@K 的柱状图对比。

### 2. 用户推荐
输入用户 ID，查看不同模型为该用户生成的个性化推荐列表，支持：
- Popularity（热门推荐）
- UserCF（相似用户）
- ItemCF（物品协同）
- NCF（深度学习）

### 3. 可解释推荐
查看推荐结果的解释，提供三种解释方案：

| 解释类型 | 方法 | 输出示例 |
|---------|------|---------|
| 关键词重叠 | 提取用户历史评论与物品评论的共同关键词 | `"You frequently mention: quality, price, battery. This product's key words: battery, durable, quality. Overlap: battery, quality."` |
| 相似用户 | 找出评分过该物品的相似用户 | `"3 similar users who bought this item also liked: ..."` |
| 时间序列 | 展示该物品的评分随时间变化趋势 | 评分趋势折线图 |

### 4. Embedding 可视化
使用 t-SNE 降维展示用户和物品的 Embedding 分布，直观观察模型的表示学习效果。

---

## 配置说明

所有配置集中在 [`config.py`](config.py)，通过字典统一管理。

### 数据配置

```python
CONFIG = {
    "data": {
        "raw_file": "data/raw/Electronics_5.json.gz",
        "output_dir": "data/processed",
    },
    "filter": {
        "k_core": 5,                 # k-core 过滤阈值
        "n_users_sample": 5000,      # 最大采样用户数
        "n_items_target": 3000,      # 目标物品数
    },
}
```

### 模型配置

```python
CONFIG = {
    "model": {
        "embedding_dim": 64,          # Embedding 维度
        "ncf_mlp_layers": [64, 32, 16],  # MLP 层结构
        "lightgcn_layers": 3,         # LightGCN 层数
        "fusion_alpha": 0.3,          # 文本语义融合权重
        "learning_rate": 0.001,
        "batch_size": 512,
        "epochs": 50,
        "early_stop_patience": 5,     # 早停耐心值
    },
    "negative_sampling": {
        "neg_ratio": 4,               # 负采样比例
        "resample_per_epoch": True,   # 每轮重采样
    },
}
```

### 评估配置

```python
CONFIG = {
    "eval": {
        "top_k": [5, 10, 20],         # 评估 K 值
    },
    "seed": 42,                       # 全局随机种子
}
```

---

## 实验结果与解读

### 评估指标

| 指标 | 全称 | 说明 |
|------|------|------|
| **Recall@K** | 召回率 | 推荐列表中命中的相关物品数 / 用户所有相关物品数 |
| **NDCG@K** | 归一化折损累计增益 | 考虑排序位置的加权召回，越靠前权重越高 |
| **Precision@K** | 精确率 | 推荐列表中命中的相关物品数 / K |
| **HitRate@K** | 命中率 | 推荐列表中是否至少包含一个相关物品 |
| **MAP@K** | 平均精度均值 | 所有用户 Average Precision 的平均值 |

### 预期实验结论

```
模型排名（预期）：
Popularity < UserCF ≈ ItemCF < NCF ≈ NCF+BPR < NCF+Review < NCF+Temporal < LightGCN ≈ Ensemble

解释：
- Popularity 作为最低基线，无个性化
- UserCF/ItemCF 利用协同信号，有个性化
- NCF 用深度网络学习交互模式，优于传统方法
- NCF+Review 利用语义信息进一步提升
- NCF+Temporal 关注近期行为，对时序数据更敏感
- LightGCN 利用高阶图结构信息，理论上界最高
- Ensemble 融合不同模型优势，效果最稳定
```

> 具体数值因数据集和超参而异，实际结果以实验输出为准。

---

## 常见问题排查

### 数据下载失败

```powershell
# 手动下载后放入 data/raw/ 目录
# 文件必须命名为 Electronics_5.json.gz 或 Electronics_5.json
```

### CUDA / GPU 相关问题

```powershell
# 查看 PyTorch 是否识别 GPU
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，使用 CPU 训练（较慢但功能不受影响）
# 自动降级，无需手动配置
```

### 内存不足 (OOM)

```powershell
# 处理大数据集时，减少采样用户数
# 修改 config.py 中的 n_users_sample 为 2000
```

### SBERT 下载超时

```powershell
# 设置镜像源（在 config.py 中已配置 hf-mirror.com）
# 或在命令行设置环境变量：
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### Streamlit 应用无法启动

```powershell
# 确保已安装所有依赖
pip install -r requirements.txt

# 确认预处理已完成
python -c "from pathlib import Path; import json; print(Path('data/processed/stats.json').exists())"
```

---

## 贡献指南

欢迎对本项目提出改进建议。请遵循以下流程：

1. **Fork** 本仓库
2. **创建特性分支** (`git checkout -b feature/your-feature`)
3. **提交更改** (`git commit -m 'Add your feature'`)
4. **推送到分支** (`git push origin feature/your-feature`)
5. **创建 Pull Request**

### 代码规范

- Python 代码遵循 [PEP 8](https://peps.python.org/pep-0008/) 规范
- 实验脚本使用 argparse 解析命令行参数
- 新增实验需在 `run_all.py` 的 EXPERIMENTS 字典中注册

### 测试

```powershell
# 运行全部测试
python -m pytest tests/

# 运行单个测试
python -m pytest tests/test_model_loading.py
```

---

## 许可证

本项目基于 MIT 许可证开源。

---

## 参考文献

1. **NCF**: He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). *Neural Collaborative Filtering*. WWW 2017. [arXiv:1708.05031](https://arxiv.org/abs/1708.05031)
2. **LightGCN**: He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. SIGIR 2020. [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)
3. **SBERT**: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
4. **BPR**: Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback*. UAI 2009.
5. **Amazon Dataset**: McAuley, J., Targett, C., Shi, Q., & Van Den Hengel, A. (2015). *Image-Based Recommendations on Styles and Substitutes*. SIGIR 2015.