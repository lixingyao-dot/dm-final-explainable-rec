# 实验命令速查
选择如下配置，不容易出问题:PyTorch 2.0.1 + CUDA 11.8 镜像，Python 3.10 开箱自带
下载下面包即可
pip install sentence-transformers shap streamlit plotly wordcloud nltk packaging transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

pip uninstall sentence-transformers -y && pip install sentence-transformers transformers==4.41.2 -i https://pypi.tuna.tsinghua.edu.cn/simple


## 预处理
数据集如果下载不了可以本地下载后上传
```bash
# 小数据集 (5K 用户 × 3K 物品)
python src/preprocess.py --users 5000 --items 3000

# 大数据集 (20K 用户 × 10K 物品)
python src/preprocess.py --users 20000 --items 10000
```
确保processed目录下有处理后的数据


实验依次执行下面的命令即可 不过为了并行可以分工
大数据集
运行 python src/preprocess.py --users 20000 --items 10000
zhx  实验0 实验1 实验1b 实验2 实验3 
gy 实验0 实验1 实验4 实验5

小数据集
运行python src/preprocess.py --users 20000 --items 10000
zyh 实验0 实验1 实验1b 实验2 实验3 
ygc 实验0 实验1 实验4 实验5







## 各实验单独运行
### 实验 0 — NCF 超参搜索

```bash
python experiments/exp0_hyper_search.py
```

### 实验 1 — 基线对比
```bash
python experiments/exp1_baseline.py --models popularity usercf itemcf --sampled
```

### 实验 1b — NCF BCE vs BPR
```bash
python experiments/exp1b_ncf_loss.py --models ncfbpr --train
```

### 实验 2 — 语义贡献（可选，已证明语义无用）
```bash
python experiments/exp2_semantic.py --sampled
```

### 实验 3 — 图模型（可选，已证明 LightGCN 不敌 BCE）
```bash
python experiments/exp3_graph.py --models lightgcn --train --sampled
```

### 实验 4 — 时序衰减
```bash
python experiments/exp4_temporal.py --models temporal --train
```

### 实验 5 — NCF + ItemCF 融合
```bash
python experiments/exp5_ensemble.py
```

