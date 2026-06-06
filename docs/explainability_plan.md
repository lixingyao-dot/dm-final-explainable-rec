# 可解释性模块 + 端到端推荐系统实施计划

## 主题定位

**时序感知的协同过滤推荐方法研究** — 推荐引擎 NCF+Temporal（创新），解释引擎 ItemCF + 时序权重归因（可解释）。

---

## 推荐与解释的整体逻辑

```
用户请求
  ↓
NCF+Temporal 融合模型 (时序衰减训练)
  ├── NCF backbone → 深度行为建模
  ├── Temporal decay → 近期行为加权 (创新点)
  └── 输出 Top-K 推荐
  ↓
解释层 (每条推荐附带)
  ├── ItemCF 协同推理 → "买过 A、B 的人通常也买这个"
  └── 时序权重归因 → "最近 N 天的行为对推荐影响最大"
```

解释和推荐共享同一个逻辑：推荐来自模型评分，解释来自评分中可追溯的部分。不是先推荐再硬编理由。

---

## 实施步骤

### 第1步：ItemCF 加 explain() 方法

**文件**：[src/base_model/itemcf.py](src/base_model/itemcf.py)

新增 `explain(user_id, item_id, k=5)` 方法：

1. 查 `self.similarities[item_id]`，找最相似的 k 个物品
2. 查用户历史交互，取其与相似物品的交集作为"桥梁物品"
3. 返回 dict：

```python
{
    "recommended_item": 1337,
    "bridge_items": [
        {"item_id": 521, "similarity": 0.87},
        {"item_id": 888, "similarity": 0.74}
    ],
    "explanation": "推荐该商品是因为你购买过 521、888，而购买这些商品的用户通常也购买 1337"
}
```

代码量：~30 行。

---

### 第2步：时序权重归因解释

**新文件**：[src/explain.py](src/explain.py) 中新增 `TemporalExplainer` 类

给一个用户和推荐商品，回查训练数据中该用户的历史记录，用 `compute_time_weights()` 计算每条历史的时间权重，按权重从高到低排序，取前 k 条作为解释。

返回 dict：

```python
{
    "top_weighted_items": [
        {"item_id": 521, "days_ago": 3,  "weight": 0.98},
        {"item_id": 888, "days_ago": 7,  "weight": 0.85},
        {"item_id": 102, "days_ago": 30, "weight": 0.35}
    ],
    "explanation": "推荐主要参考最近行为：Sony WH-1000XM4 (3天前, 权重0.98) > Bose QC45 (7天前, 权重0.85)"
}
```

代码量：~50 行。

---

### 第3步：批量解释生成脚本

**新文件**：[experiments/exp6_explain.py](experiments/exp6_explain.py)

流程：
1. 加载数据（train.csv、test.csv、stats.json）
2. 加载 ItemCF 模型
3. 对测试集中每个用户，用 ItemCF 推荐 1 个商品
4. 调用 ItemCF.explain() 和 TemporalExplainer 生成双层解释
5. 输出 `outputs/exp6_explanations.json`

代码量：~80 行。

---

### 第4步：Streamlit Web Demo

**新文件**：[app/streamlit_app.py](app/streamlit_app.py)

3 个页面：

**页面 1 — 推荐解释（主界面）**
- 下拉选择用户 ID
- 展示 Top-10 推荐列表（ItemCF 评分）
- 点击某条推荐 → 展开解释卡片：
  - 协同推理："因为买过 X、Y..."
  - 时序归因："最近行为权重排名"

**页面 2 — 模型对比**
- 表格 + 柱状图：所有模型 H@10、NDCG@10 对比
- 标注 NCF+Temporal 最高分（+15%）
- 标注 ItemCF 作为可解释基线

**页面 3 — 时序衰减分析**
- λ 从 0.5 到 5.0 的 H@10 变化折线图
- 直观展示"越近的行为越重要"

代码量：~200 行。

---

## 工作量汇总

| 步骤 | 内容 | 代码量 | 依赖 |
|------|------|--------|------|
| 1 | ItemCF explain() | ~30行 | self.similarities 已存在 |
| 2 | TemporalExplainer | ~50行 | compute_time_weights 已存在 |
| 3 | exp6_explain.py | ~80行 | 步骤1+2 |
| 4 | streamlit_app.py | ~200行 | streamlit, plotly |
| **合计** | | **~360行** | |

---

## 答辩呈现

1. **现场演示**：打开 Web 页面，选用户 → 推荐列表 → 点击展开解释
2. **PPT 实验表**：NCF+Temporal H@10=0.392，ItemCF 0.339，用 0.34 的解释成本换取 15% 准确率提升
3. **PPT 案例截图**：一个用户的完整推荐解释（协同推理 + 时序归因）
4. **PPT 时序曲线图**：λ↑ → H@10↑，核心发现一图胜千言
