# 模块B 详细实施计划

> 制定日期：2026-05-28
>
> 前置条件：模块A（数据管道）和模块D（SBERT 编码 + 可解释模块）已成功完成并通过验收

---

## 一、前期评估：项目当前状态

### ✅ 已完成（模块A + D，已验收通过）

| 交付件 | 文件 | 说明 |
|-------|------|------|
| 数据管道 Stage 1-8 | `src/preprocess.py` | 端到端 Pipeline：JSONL 流式加载 → K-core → 采样 → LOO 划分 → 负采样 → 文本聚合 → SBERT 编码 |
| 工具函数 | `src/utils.py` | 断点续跑、文本清洗、负采样、聚合截断 |
| 配置中心 | `config.py` | 全部参数集中管理 |
| 可解释模块 | `src/explain.py` | KeywordExplainer、SimilarUserExplainer、SHAPExplainer |

### 📦 可供模块B直接使用的产出（来自A+D）

| 数据文件 | 用途 |
|---------|------|
| `data/processed/train.csv` | NCF/NCF+Review 训练（正样本） |
| `data/processed/val.csv` | 早停 + 调参 |
| `data/processed/test.csv` | 最终评估（仅跑一次） |
| `data/processed/stats.json` | n_users, n_items |
| `data/processed/user_emb.npy` | SBERT 用户语义向量（n_users × 384），NCF+Review 融合用 |
| `data/processed/item_emb.npy` | SBERT 商品语义向量（n_items × 384），NCF+Review 融合用 |

### ✅ 已有但需模块B验证/运行的代码

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/ncf.py` | 结构完成 | NCF 模型类 + 训练循环（GMF+MLP 双通路融合） |
| `src/ncf_review.py` | 结构完成 | NCF+Review 融合模型（behavior + semantic 加权融合） |
| `src/evaluate.py` | 可直接使用 | Precision@K, Recall@K, HitRate, MAP, NDCG |
| `experiments/exp1_baseline.py` | 含 4 个模型 | Popularity + UserCF + ItemCF + NCF（不含 NCF+Review） |
| `src/baselines.py` | 可直接使用 | PopularityRecommender, UserCF, ItemCF |

### ❌ 模块B需要完成的工作

| 缺失项 | 优先级 | 说明 |
|--------|--------|------|
| 实验2脚本：NCF vs NCF+Review | 🔴 核心 | 调用 SBERT embedding，做语义贡献对比 |
| Alpha 消融实验 | 🟡 重要 | α = 0.1 / 0.3 / 0.5 三组对比 |
| NCF 实际训练与调参 | 🔴 核心 | 当前只有代码骨架，需要跑出指标 |
| NCF+Review 实际训练调参 | 🔴 核心 | 同上，需对接 D 的 embedding |
| 结果持久化 | 🟡 重要 | 每轮实验自动写入 JSON + CSV |
| 报告模型章节 | 🔴 核心 | NCF 设计、NCF+Review 融合设计、实验2讨论 |
| PPT 对应部分 | 🟢 次要 | 第2-3周集中做 |

---

## 二、模块B任务范围正式界定

从项目路线图第一版分工表（最新版），模块B的职责明确为：

> **B = NCF + NCF+Review（不含 LightGCN）**

对应三个研究问题中的 **RQ2：评论语义能不能提升推荐**。

---

## 三、详细实施步骤（按天拆分）

### 第一阶段：NCF 训练与调参（Day 1-3）

#### Day 1：环境验证 + NCF 首次训练

| 任务 | 具体内容 | 预期产出 |
|------|---------|---------|
| 1.1 确认数据就绪 | 检查 `data/processed/` 下 train/val/test/stats 是否存在 | 确认可读 |
| 1.2 运行 NCF 首次训练 | 运行临时脚本或 `exp1_baseline.py` 中的 NCF 部分 | 50 epoch 训练完成 |
| 1.3 评估 NCF | 调用 `evaluate_model(ncf_model, test_df, train_df, n_items)` | NCF 首版指标 |
| 1.4 记录结果 | 保存到 `outputs/ncf_v1.json` | v1 基线 |

**涉及的代码文件**：`src/ncf.py`、`src/evaluate.py`

#### Day 2：NCF 超参数调优

| 参数 | 建议搜索范围 | 固定值 |
|------|-------------|-------|
| `embedding_dim` | 32, **64**, 128 | — |
| `learning_rate` | **0.001**, 0.0005, 0.005 | — |
| `mlp_layers` | **[64,32,16]**, [128,64,32], [64,64] | — |
| `batch_size` | **256**, 512, 1024 | — |
| `neg_ratio` | **4**（已在 config 固定） | 4 |
| `early_stop_patience` | 5（已在 config 固定） | 5 |

**方式**：手动跑 6-8 组组合（非网格搜索），每组 20-30 epoch 用 val_loss 早停。

**产出**：`outputs/ncf_hyper/` 下若干 JSON，记录每组参数与指标。

#### Day 3：NCF 最优模型 + 完整评估

| 任务 | 内容 |
|------|------|
| 3.1 选定最优超参 | 根据 Day 2 val 指标选最优 |
| 3.2 全量训练（50 epoch） | 用最优参数重训一次 |
| 3.3 完整 test 评估 | Precision/Recall/HitRate/MAP/NDCG @ 5/10/20 |
| 3.4 保存模型权重 | `outputs/models/ncf_best.pt` |

> **注意**：NCF 训练时每 epoch 重采负样本（`resample_per_epoch=True`），LightGCN 的 BPR 策略不同，两套机制互不影响。

---

### 第二阶段：NCF+Review 融合训练与调参（Day 4-6）

#### Day 4：验证 SBERT Embedding + 首次 NCF+Review 训练

| 任务 | 具体内容 | 涉及文件 |
|------|---------|---------|
| 4.1 验证 embedding 可用 | 加载 `user_emb.npy`, `item_emb.npy` 确认 shape, NaN, 非零行 | — |
| 4.2 打通 NCF+Review 接口 | 确认 `NCFReview` 可接收 SBERT embedding 作为第二个输入流 | `src/ncf_review.py` |
| 4.3 首次训练（α=0.3） | 用 `train_ncf_review()` 跑 50 epoch | 首次 NCF+Review 指标 |

**关键数据流**：

```
train.csv (user_id, item_id)  ──→ NCF path (ID embedding → MLP → behavior_vec)
                                           ↓
                                  加权融合: (1-α) * behavior + α * semantic
                                           ↓
SBERT embedding ──→ Linear Proj ──→ semantic_vec ──→ Fusion MLP → score
(user_emb.npy / item_emb.npy)
```

#### Day 5：融合权重 α 消融实验（RQ2 核心）

| α 值 | 语义占比 | 实验编号 |
|-------|---------|---------|
| 0.0 | 纯行为（= NCF 基线） | exp2a |
| 0.1 | 10% 语义 | exp2b |
| **0.3** | **30% 语义** | **exp2c** |
| 0.5 | 50% 语义 | exp2d |

**产出**：`outputs/ncf_review_alpha.json` — 4 组 α 的完整 test 指标。

#### Day 6：NCF+Review 最优模型确定

| 任务 | 内容 |
|------|------|
| 6.1 选最优 α | 以 NDCG@10 + Recall@10 为判据 |
| 6.2 保存最佳模型 | `outputs/models/ncf_review_best.pt` |
| 6.3 汇总 NCF vs NCF+Review 对比表 | 准备实验2核心数据 |

---

### 第三阶段：实验2 完整运行 + 结果整理（Day 7-8）

#### Day 7：编写实验2脚本 `experiments/exp2_semantic.py`

需要完成的功能：

```
experiments/exp2_semantic.py
├── 加载 train/val/test + stats.json
├── 加载 SBERT embedding（来自 D）
├── 训练+评估 NCF（baseline）
├── 训练+评估 NCF+Review（α=0.1, 0.3, 0.5）
├── 打印对比表格（参考 exp1_baseline.py 的 summary 格式）
└── 存结果 → outputs/exp2_semantic_results.json
```

**对比表格式**：

```
  NDCG@10:              NCF      NCF+R(α=0.1)  NCF+R(α=0.3)  NCF+R(α=0.5)
  ─────────────────────────────────────────────────────────────────────
  Precision@5           0.xxxx    0.xxxx         0.xxxx         0.xxxx
  Recall@10             0.xxxx    0.xxxx         0.xxxx         0.xxxx
  NDCG@10               0.xxxx    0.xxxx         0.xxxx         0.xxxx
  ...
```

#### Day 8：补充实验 + 交叉验证稳定性

| 任务 | 内容 |
|------|------|
| 8.1 不同随机种子验证 | 用 seed=42, 123, 456 各跑一次 NCF vs NCF+Review（α=0.3） |
| 8.2 指标波动幅度 | 计算 mean ± std |
| 8.3 追加实验 | 如需，测试 review_emb_dim=384 外的降维版本（128/64） |

---

### 第四阶段：结果汇总 + 模型章节报告（Day 9-11）

#### Day 9：全量结果整合

| 任务 | 文件 |
|------|------|
| 将 NCF 最优指标纳入实验1总表 | `outputs/exp1_baseline_results.json`（更新） |
| 实验2完整结果 | `outputs/exp2_semantic_results.json` |
| 实验3对比数据（NCF vs LightGCN，与C协作） | `outputs/exp3_graph_results.json` |

#### Day 10-11：报告模型章节撰写

模块B负责的报告章节：

| 章节 | 内容要点 | 篇幅 |
|------|---------|------|
| 3.1 NCF 模型设计 | GMF+MLP 双通路、嵌入层初始化、输出层设计 | ~1页 |
| 3.2 NCF+Review 融合设计 | 行为-语义双流架构、加权融合公式、投影层 | ~1页 |
| 3.3 训练策略 | 负采样、每 epoch 重采、早停、超参数 | ~0.5页 |
| 4.2 实验2：语义贡献分析 | 对比表、α 消融分析、结论：语义是否有效 | ~1.5页 |
| 4.3 综合分析 | 所有 6 个模型总表 + 讨论（与A/C协作） | ~1页 |

---

### 第五阶段：PPT + 交叉校对（Day 12-14）

#### Day 12：PPT 制作

| 页 | 内容 |
|----|------|
| NCF 架构图 | GMF+MLP 双通路示意图 |
| NCF+Review 融合图 | 行为-语义双流 + 加权融合 |
| 实验2结果表 | NCF vs NCF+Review @ 不同 α |
| 关键结论 | 语义贡献的定量结论（加分/不加分） |

#### Day 13-14：交叉校对 + 全员整合

- 配合 A 核对数据描述一致性（n_users / n_items / sparsity）
- 配合 C 核对 NCF vs LightGCN 实验3数据
- 配合 D 确认 embedding 版本一致
- 全报告格式统一

---

## 四、依赖关系与前置条件

### 外部依赖

| 依赖模块 | 需要什么 | 当前状态 |
|---------|---------|---------|
| A（数据管道） | `train.csv`, `val.csv`, `test.csv`, `stats.json` | ✅ 已验收 |
| D（SBERT） | `user_emb.npy`, `item_emb.npy`（(n_users×384), (n_items×384)） | ✅ 已验收 |

### 内部依赖（模块B内部）

```
Day1-3 NCF训练调参 ──→ Day4-6 NCF+Review融合 ──→ Day7-8 实验2脚本 ──→ Day9-11 报告
```

✅ NCF 和 NCF+Review **互不阻塞**，NCF 调优经验可直接复用到 NCF+Review 的训练策略上。

---

## 五、交付物清单

| 序号 | 交付物 | 文件路径 | 预期截止 |
|------|-------|---------|---------|
| 1 | NCF 最优模型权重 | `outputs/models/ncf_best.pt` | Day 3 |
| 2 | NCF+Review 最优模型权重 | `outputs/models/ncf_review_best.pt` | Day 6 |
| 3 | 实验2脚本 | `experiments/exp2_semantic.py` | Day 7 |
| 4 | 实验2完整结果 | `outputs/exp2_semantic_results.json` | Day 8 |
| 5 | Alpha 消融结果 | `outputs/ncf_review_alpha.json` | Day 5 |
| 6 | 实验1总表（NCF 补入） | `outputs/exp1_baseline_results.json`（更新版） | Day 9 |
| 7 | 报告模型章节 | 写入 report doc | Day 11 |
| 8 | PPT 对应页 | PPT | Day 12 |

---

## 六、风险与应对

| 风险 | 概率 | 影响 | 应对方案 |
|------|------|------|---------|
| NCF+Review 未涨点（语义反而引入噪音） | 中 | RQ2 结论不好看 | 分析失败原因：①embedding 质量 ②融合位置不对 ③尝试 late fusion 替代 weighted sum |
| SBERT embedding 需要降维（384 维过大） | 低 | 训练速度 | 在 NCF+Review 的 projection 层已处理（384→64），无需额外降维 |
| NCF 训练不收敛 | 低 | 浪费时间 | 检查：①embedding init ②learning rate ③负采样 ratio ④数据稀疏度 |
| 与 C 的 LightGCN 对比数据不一致 | 低 | 报告矛盾 | 统一用 `evaluate_model()` 函数，确保完全相同 test set 和 metric |

---

## 七、与模块C的协作接口

虽然模块B和C是并行关系，但实验3（NCF vs LightGCN）需要两者的指标。

| 共享内容 | B 提供 | C 提供 |
|---------|-------|-------|
| 实验3 对比表 | NCF 在 test set 上的全部指标 | LightGCN 在 test set 上的全部指标 |
| 统一指标函数 | `from src.evaluate import evaluate_model, print_metrics` | 同上 |
| 统一评估配置 | top_k=[5, 10, 20] | 同上 |

> **关键规则**：确保实验3 中 NCF 和 LightGCN 使用**完全相同的 test set**（`data/processed/test.csv`），否则对比无效。

---

## 八、验收标准

模块B通过验收的条件：

| # | 标准 | 验证方式 |
|---|------|---------|
| 1 | NCF 能在 test set 上跑出完整的 Precision/Recall/HitRate/MAP/NDCG @ 5/10/20 | 运行 `exp2_semantic.py` |
| 2 | NCF+Review 能正常加载 SBERT embedding 并完成融合训练 | 同上 |
| 3 | 实验2 输出 NCF vs NCF+Review（≥2 种 α）对比表 | 检查 `exp2_semantic_results.json` |
| 4 | 所有结果可复现（固定 seed） | 重新运行与记录一致 |
| 5 | 模型权重和结果文件均已持久化 | 检查 outputs/ 目录 |
| 6 | 报告模型章节初稿完成 | 审查文档 |