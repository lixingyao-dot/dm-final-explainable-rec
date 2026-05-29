"""NCF variants head-to-head comparison."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CONFIG as BASE_CONFIG
from config_improved import CONFIG as IMPROVED_CONFIG
from src.ncf import NCF, train_ncf
from src.ncf_bias import NCFBias, train_ncf_bias
from src.ncf_bpr import NCFBPR, train_ncf_bpr
from src.ncf_bpr_bias import NCFBiasBPR, train_ncf_bpr_bias
from src.evaluate import evaluate_model_sampled, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["ncfbias"],
                        choices=["ncf", "ncfbias", "ncfbpr", "ncfbprbias", "all"])
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    run_models = set(args.models)
    if "all" in run_models:
        run_models = {"ncf", "ncfbias", "ncfbpr", "ncfbprbias"}

    data_dir = Path(BASE_CONFIG["data"]["output_dir"])
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    with open(data_dir / "stats.json") as f:
        stats = json.load(f)
    n_users, n_items = stats["n_users"], stats["n_items"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Models: {sorted(run_models)}")

    all_results = {}

    if "ncf" in run_models:
        print("\n" + "=" * 60)
        print("  Training NCF (original, dim=64, neg=4, no bias)")
        print("=" * 60)
        t0 = time.time()
        model = NCF(
            n_users=n_users, n_items=n_items,
            embedding_dim=BASE_CONFIG["model"]["embedding_dim"],
            mlp_layers=BASE_CONFIG["model"]["ncf_mlp_layers"],
        ).to(device)

        ckpt_path = Path("outputs/models/ncf_best.pt")
        if not args.train and ckpt_path.exists():
            print("  Loading checkpoint ...")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            for name, param in model.named_parameters():
                if name in state and state[name].shape != param.shape:
                    old_shape = state[name].shape
                    new = torch.zeros_like(param)
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, param.shape))
                    new[slices] = state[name][slices]
                    state[name] = new
            model.load_state_dict(state)
        else:
            model = train_ncf(model, train_df, val_df, BASE_CONFIG, n_items=n_items, device=device)

        all_results["NCF_original"] = evaluate_model_sampled(model, test_df, train_df, n_items)
        print_metrics(all_results["NCF_original"], "NCF (dim=64, neg=4)")
        torch.save(model.state_dict(), "outputs/models/ncf_best.pt")
        print(f"  Time: {time.time() - t0:.1f}s")

    if "ncfbias" in run_models:
        print("\n" + "=" * 60)
        print("  Training NCFBias (dim=32, neg=9, +user/item bias)")
        print("=" * 60)
        t0 = time.time()
        model = NCFBias(
            n_users=n_users, n_items=n_items,
            embedding_dim=IMPROVED_CONFIG["model"]["embedding_dim"],
            mlp_layers=IMPROVED_CONFIG["model"]["ncf_mlp_layers"],
        ).to(device)

        ckpt_path = Path("outputs/models/ncf_bias_best.pt")
        if not args.train and ckpt_path.exists():
            print("  Loading checkpoint ...")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            for name, param in model.named_parameters():
                if name in state and state[name].shape != param.shape:
                    old_shape = state[name].shape
                    new = torch.zeros_like(param)
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, param.shape))
                    new[slices] = state[name][slices]
                    state[name] = new
            model.load_state_dict(state)
        else:
            model = train_ncf_bias(model, train_df, val_df, IMPROVED_CONFIG, n_items=n_items, device=device)

        all_results["NCFBias"] = evaluate_model_sampled(model, test_df, train_df, n_items)
        print_metrics(all_results["NCFBias"], "NCFBias (dim=32, neg=9, bias)")
        torch.save(model.state_dict(), "outputs/models/ncf_bias_best.pt")
        print(f"  Time: {time.time() - t0:.1f}s")

    if "ncfbpr" in run_models:
        print("\n" + "=" * 60)
        print("  Training NCF+BPR (dim=64, BPR pairwise loss)")
        print("=" * 60)
        t0 = time.time()
        model = NCFBPR(
            n_users=n_users, n_items=n_items,
            embedding_dim=BASE_CONFIG["model"]["embedding_dim"],
            mlp_layers=BASE_CONFIG["model"]["ncf_mlp_layers"],
        ).to(device)

        ckpt_path = Path("outputs/models/ncf_bpr_best.pt")
        if not args.train and ckpt_path.exists():
            print("  Loading checkpoint ...")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            for name, param in model.named_parameters():
                if name in state and state[name].shape != param.shape:
                    old_shape = state[name].shape
                    new = torch.zeros_like(param)
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, param.shape))
                    new[slices] = state[name][slices]
                    state[name] = new
            model.load_state_dict(state)
        else:
            model = train_ncf_bpr(model, train_df, val_df, BASE_CONFIG, n_items=n_items, device=device)

        all_results["NCF_BPR"] = evaluate_model_sampled(model, test_df, train_df, n_items)
        print_metrics(all_results["NCF_BPR"], "NCF+BPR")
        torch.save(model.state_dict(), "outputs/models/ncf_bpr_best.pt")
        print(f"  Time: {time.time() - t0:.1f}s")

    if "ncfbprbias" in run_models:
        print("\n" + "=" * 60)
        print("  Training NCFBias+BPR (dim=32, neg=9, bias + BPR)")
        print("=" * 60)
        t0 = time.time()
        model = NCFBiasBPR(
            n_users=n_users, n_items=n_items,
            embedding_dim=IMPROVED_CONFIG["model"]["embedding_dim"],
            mlp_layers=IMPROVED_CONFIG["model"]["ncf_mlp_layers"],
        ).to(device)

        ckpt_path = Path("outputs/models/ncf_bpr_bias_best.pt")
        if not args.train and ckpt_path.exists():
            print("  Loading checkpoint ...")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            for name, param in model.named_parameters():
                if name in state and state[name].shape != param.shape:
                    old_shape = state[name].shape
                    new = torch.zeros_like(param)
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, param.shape))
                    new[slices] = state[name][slices]
                    state[name] = new
            model.load_state_dict(state)
        else:
            model = train_ncf_bpr_bias(model, train_df, val_df, IMPROVED_CONFIG, n_items=n_items, device=device)

        all_results["NCFBiasBPR"] = evaluate_model_sampled(model, test_df, train_df, n_items)
        print_metrics(all_results["NCFBiasBPR"], "NCFBias+BPR (dim=32, neg=9, bias)")
        torch.save(model.state_dict(), "outputs/models/ncf_bpr_bias_best.pt")
        print(f"  Time: {time.time() - t0:.1f}s")

    # Summary
    print("\n" + "=" * 60)
    print("  NCF vs NCFBias COMPARISON")
    print("=" * 60)
    first = next(iter(all_results.values()))
    k_vals = sorted(set(int(k.split("@")[1]) for k in first.keys()))
    for metric in ["HitRate", "NDCG"]:
        print(f"\n  {metric}:")
        header = f"  {'Model':<18}" + "".join(f"{'@' + str(k):<12}" for k in k_vals)
        print(header)
        print(f"  {'─' * (18 + 12 * len(k_vals))}")
        for name, res in all_results.items():
            row = f"  {name:<18}" + "".join(f"{res.get(f'{metric}@{k}', 0):<12.4f}" for k in k_vals)
            print(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("outputs") / f"exp1_bias_{timestamp}.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
