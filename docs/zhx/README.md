# zhx 实验脚本说明

## 负责实验（大数据集）

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 预处理 | `src/preprocess.py --users 20000 --items 10000` | 大数据集预处理 |
| 实验 0 | `exp0_hyper_search.py` | NCF 超参搜索 |
| 实验 1 | `exp1_baseline.py --models popularity usercf itemcf --sampled` | 基线对比（大数据集启用 --sampled） |
| 实验 1b | `exp1b_ncf_loss.py --models ncfbpr --train` | NCF BCE vs BPR 损失函数对比 |
| 实验 2 | `exp2_semantic.py --sampled` | 语义贡献分析 |
| 实验 3 | `exp3_graph.py --models lightgcn --train --sampled` | 图模型 LightGCN |

## 使用方法

```bash
# 进入脚本目录
cd docs/zhx

# 后台挂起运行
nohup bash run_zhx.sh > runner.log 2>&1 &

# 查看总体进度
tail -f runner.log

# 查看各实验详细日志（已过滤 tqdm 每轮进度条）
tail -f logs/preprocess.log
tail -f logs/exp0.log
tail -f logs/exp1.log
tail -f logs/exp1b.log
tail -f logs/exp2.log
tail -f logs/exp3.log

# 检查所有实验是否完成
grep "完成" runner.log
```

## 日志说明

- `logs/preprocess.log` — 预处理日志
- `logs/exp<N>.log` — 各实验独立日志，已过滤掉含 `Epoch` 的行（tqdm 进度条）
- `runner.log` — 脚本运行总体进度（各步骤开始/结束时间、退出码）

## 环境要求

参考项目根目录 `requirements-gpu-compat.txt` 安装依赖：

```bash
pip install -r requirements-gpu-compat.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```