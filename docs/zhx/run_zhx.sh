#!/bin/bash
# ============================================================
# zhx 负责的实验（大数据集）：实验0 / 1 / 1b / 2 / 3
# 用法：
#   nohup bash run_zhx.sh &          # 后台挂起运行
#   tail -f logs/runner.log          # 查看总体进度
#   tail -f logs/exp<N>.log          # 查看单个实验日志（已过滤 tqdm 每轮进度条）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"
echo "工作目录: $(pwd)"
echo "日志目录: $LOG_DIR"

# ── 日志过滤器 ──────────────────────────────────────────────
# 去掉含 Epoch 的行（tqdm 进度条），保留其他所有输出
filter_epoch() {
    grep -v -E 'Epoch'
}

# ── 预处理（大数据集 20K 用户 × 10K 物品）──────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 预处理: 20K 用户 × 10K 物品 ====="
python src/preprocess.py --users 20000 --items 10000 2>&1 | filter_epoch >> "$LOG_DIR/preprocess.log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 预处理完成（exit code: $?）"

# ── 依次运行实验 ────────────────────────────────────────────

# 实验 0 — NCF 超参搜索
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 实验 0: NCF 超参搜索 ====="
python experiments/exp0_hyper_search.py 2>&1 | filter_epoch >> "$LOG_DIR/exp0.log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验 0 完成（exit code: $?）"

# 实验 1 — 基线对比
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 实验 1: 基线对比 ====="
python experiments/exp1_baseline.py --models popularity usercf itemcf --sampled 2>&1 | filter_epoch >> "$LOG_DIR/exp1.log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验 1 完成（exit code: $?）"

# 实验 1b — NCF BCE vs BPR
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 实验 1b: NCF BCE vs BPR ====="
python experiments/exp1b_ncf_loss.py --models ncfbpr --train 2>&1 | filter_epoch >> "$LOG_DIR/exp1b.log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验 1b 完成（exit code: $?）"

# 实验 2 — 语义贡献
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 实验 2: 语义贡献 ====="
python experiments/exp2_semantic.py --sampled 2>&1 | filter_epoch >> "$LOG_DIR/exp2.log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验 2 完成（exit code: $?）"

# 实验 3 — 图模型
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 实验 3: 图模型 (LightGCN) ====="
python experiments/exp3_graph.py --models lightgcn --train --sampled 2>&1 | filter_epoch >> "$LOG_DIR/exp3.log" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 实验 3 完成（exit code: $?）"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== 全部实验运行完毕 ====="