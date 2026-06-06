#!/bin/bash
set -e

echo "========== 预处理 =========="
python src/preprocess.py --users 20000 --items 10000

echo "========== 实验0 — NCF 超参搜索 =========="
python experiments/exp0_hyper_search.py

echo "========== 实验1 — 基线对比 =========="
python experiments/exp1_baseline.py --models popularity usercf itemcf --sampled

echo "========== 实验1b — NCF BCE vs BPR =========="
python experiments/exp1b_ncf_loss.py --models ncfbpr --train

echo "========== 实验2 — 语义贡献 =========="
python experiments/exp2_semantic.py --sampled

echo "========== 实验3 — 图模型 =========="
python experiments/exp3_graph.py --models lightgcn --train --sampled

echo "========== 实验4 — 时序衰减 =========="
python experiments/exp4_temporal.py --models temporal --train

echo "========== 实验5 — NCF + ItemCF 融合 =========="
python experiments/exp5_ensemble.py

echo "========== 全部完成 =========="
