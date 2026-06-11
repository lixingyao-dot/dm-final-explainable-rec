#!/usr/bin/env python3
"""一键运行所有实验的入口脚本。

用法：
    python run_all.py                  # 按顺序运行所有实验
    python run_all.py --exp 0          # 只运行实验 0（超参数搜索）
    python run_all.py --exp 1          # 只运行实验 1（基线对比）
    python run_all.py --exp 2          # 只运行实验 2（语义贡献）
    python run_all.py --exp 3          # 只运行实验 3（图模型对比）
    python run_all.py --exp 4          # 只运行实验 4（时间衰减）
    python run_all.py --exp 5          # 只运行实验 5（集成模型）
    python run_all.py --exp 6          # 只运行实验 6（可解释性 & TA-ItemCF）
    python run_all.py --list           # 列出所有实验
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(step_name: str, command: list, cwd: str = None):
    print(f"\n{'#' * 70}")
    print(f"#  STEP: {step_name}")
    print(f"#  CMD:  {' '.join(command)}")
    print(f"{'#' * 70}\n")
    t0 = time.time()
    result = subprocess.run(command, capture_output=False, cwd=cwd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ STEP FAILED: {step_name} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n  ✓ Step completed in {elapsed:.1f}s")
    return result


def setup_output_dirs():
    root = Path(__file__).resolve().parent
    dirs = [
        "outputs/models",
        "outputs/ncf_hyper",
        "outputs/explanations",
    ]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)


def get_project_root():
    return str(Path(__file__).resolve().parent)


def main():
    parser = argparse.ArgumentParser(description="一键运行可解释推荐系统全部实验")
    parser.add_argument("--exp", type=int, choices=range(0, 7), default=None,
                        help="运行指定实验 (0-6)。不指定则运行全部。")
    parser.add_argument("--list", action="store_true",
                        help="列出所有实验及其说明")
    args = parser.parse_args()

    # 列出实验
    if args.list:
        print("=" * 70)
        print("  可解释推荐系统 — 实验列表")
        print("=" * 70)
        for eid, (name, _) in sorted(EXPERIMENTS.items()):
            print(f"  [{eid}] {name}")
        print()
        return

    setup_output_dirs()
    root = get_project_root()

    print("=" * 70)
    print("  可解释推荐系统 — 全流程实验管线")
    print("  项目: 基于用户行为与评论文本的可解释推荐系统")
    print("=" * 70)

    if args.exp is not None:
        # 运行单个实验
        if args.exp not in EXPERIMENTS:
            print(f"未知实验编号: {args.exp}")
            print("可用实验:")
            for eid, (name, _) in sorted(EXPERIMENTS.items()):
                print(f"  [{eid}] {name}")
            sys.exit(1)
        name, cmd = EXPERIMENTS[args.exp]
        run_step(name, cmd, cwd=root)
    else:
        # 按顺序运行全部实验
        for exp_id in sorted(EXPERIMENTS.keys()):
            name, cmd = EXPERIMENTS[exp_id]
            run_step(name, cmd, cwd=root)

    print(f"\n{'=' * 70}")
    print("  所有实验已完成！")
    print(f"{'=' * 70}")
    print("\n  产出文件:")
    print("    outputs/ncf_hyper/          - 超参数搜索结果")
    print("    outputs/models/             - 训练好的模型权重")
    print("    outputs/explanations/       - 可解释性结果")
    print("\n  启动可视化界面:")
    print("    streamlit run app/streamlit_app.py")


# ── 实验注册表 ──
# 按依赖顺序排列：超参数搜索 → 基线 → 模型对比 → 高级实验 → 可解释性
EXPERIMENTS = {
    0: ("实验 0: NCF 超参数搜索 (embedding_dim × learning_rate)",
        ["python", "experiments/exp0_hyper_search.py"]),
    1: ("实验 1: 基线模型对比 (Popularity / UserCF / ItemCF / NCF)",
        ["python", "experiments/exp1_baseline.py"]),
    # exp1b 作为扩展，不自动运行。如需运行：
    #   python experiments/exp1b_ncf_loss.py --models all --train
    2: ("实验 2: 语义贡献分析 (NCF vs NCF+Review)",
        ["python", "experiments/exp2_semantic.py"]),
    3: ("实验 3: 图模型对比 (NCF vs LightGCN)",
        ["python", "experiments/exp3_graph.py"]),
    4: ("实验 4: 时间衰减实验 (NCF vs NCF+Temporal)",
        ["python", "experiments/exp4_temporal.py"]),
    5: ("实验 5: 集成模型 (NCF + ItemCF Ensemble)",
        ["python", "experiments/exp5_ensemble.py"]),
    6: ("实验 6a: 可解释性导出 (ItemCF & Temporal Explanations)",
        ["python", "experiments/exp6_explain.py"]),
    # exp6_ta_itemcf 作为扩展实验
}

if __name__ == "__main__":
    main()