"""
N-Unpairs Experiment
====================
Varies the number of unpaired training samples as a percentage of the full
training set and evaluates the total correlation on a held-out test set.

Usage:
    python exps/exp_n_unpairs.py --method sca --data handwritten
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
import json

# Ensure project root is on the path so imports work from any cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_data, prepare_data
from utils import get_total_correlation, save_results
from exps.methods import run_method_dispatch



def run_experiment(cfg: dict, config_path=None):
    """Run the N-unpairs experiment."""
    method = cfg["method"]
    data_kind = cfg["data"]
    n_repeats = cfg.get("n_repeats", 5)
    n_test = cfg.get("n_test", 256)
    seed = cfg.get("seed", 42)
    results_dir = "results/n_unpairs"
    output_size = cfg.get("output_size", 2)
    
    # We only handle scalar output_size for now in this exp
    if isinstance(output_size, list):
        print("[warn] exp_n_unpairs only supports scalar output_size. Using first element.")
        output_size = output_size[0]

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load full data
    embed_A, embed_B, labels = get_data(
        data_kind,
        n_points=cfg.get("n_points"),
        view1_idx=cfg.get("view1_idx"),
        view2_idx=cfg.get("view2_idx"),
    )
    
    # Get the "full" training and test sets
    X_train_full, X_test, Y_train_full, Y_test, L_X_train, L_Y_train, L_test = prepare_data(
        embed_A, embed_B, n_test, split=cfg.get("split", "unpaired"), labels=labels
    )
    
    n_train_full = X_train_full.shape[0]
    exp_name = cfg.get("exp_name", None)
    if exp_name is None:
        if config_path:
            exp_name = os.path.splitext(os.path.basename(config_path))[0]
        else:
            exp_name = f"{method}_{data_kind}"

    # Check for existing results (checkpointing)
    results_path = os.path.join(results_dir, f"{exp_name}.json")
    if not cfg.get("new", False) and os.path.exists(results_path):
        print(f"[checkpoint] Loading existing results from {results_path}")
        with open(results_path, "r") as f:
            final_results = json.load(f)
        results_by_pct = final_results.get("results", {})
        percentages = final_results.get("percentages", [50, 60, 70, 80, 90])
        
        # Merge with provided percentages if any
        if cfg.get("percentages_list"):
            for p in cfg["percentages_list"]:
                if p not in percentages:
                    percentages.append(p)
            percentages.sort()
            final_results["percentages"] = percentages
    else:
        results_by_pct = {}
        if not cfg.get("percentages_list"):
            percentages = [50, 60, 70, 80, 90]
        else:
            percentages = cfg["percentages_list"]
            
        final_results = {
            "method": method,
            "data": data_kind,
            "method_name": cfg.get("method_name", method),
            "config": cfg,
            "percentages": percentages,
            "results": results_by_pct
        }

    print(f"[exp] Starting/Resuming N-unpairs sweep for {method} on {data_kind}")

    def run_single_pct(pct):
        n_unpairs = int(n_train_full * pct / 100)
        print(f"\n[exp] pct={pct}% (n_unpairs={n_unpairs})")
        
        correlations = []
        for i in range(n_repeats):
            # Sample subset of training data
            idx = torch.randperm(n_train_full)[:n_unpairs]
            X_train = X_train_full[idx]
            Y_train = Y_train_full[idx]
            
            # Update config for this run, ensuring custom sweep args aren't passed
            point_cfg = cfg.copy()
            point_cfg.pop("percentages_list", None)
            point_cfg.pop("n_jobs", None)
            point_cfg.pop("new", None)

            result = run_method_dispatch(method, X_train, Y_train, X_test, Y_test, point_cfg)
            out1, out2 = result
            tc = get_total_correlation(out1, out2)
            correlations.append(tc)
            # print(f"  repeat {i + 1}/{n_repeats}: tc = {tc:.4f}")
            
        mu = float(np.mean(correlations))
        std = float(np.std(correlations))
        
        result_entry = {
            "correlations": correlations,
            "mean": mu,
            "std": std,
            "n_unpairs": n_unpairs
        }
        
        results_by_pct[str(pct)] = result_entry
        final_results["results"] = results_by_pct
        save_results(exp_name, final_results, results_dir)
        print(f"[result] pct={pct}%: tc = {mu:.4f} +/- {std:.4f}")
        
        return pct, result_entry

    to_run = [pct for pct in percentages if str(pct) not in results_by_pct]
    
    if to_run:
        for pct in to_run:
            run_single_pct(pct)

    return final_results


def main():
    parser = argparse.ArgumentParser(description="N-Unpairs Experiment")
    parser.add_argument("--method", type=str, default="ucca")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--percentages", type=str, default=None, help="Comma-separated list of percentages")
    parser.add_argument("--new", action="store_true", help="Start new experiment")
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default",
            f"{args.method}_{args.data}.yaml"
        )

    if not os.path.exists(config_path):
        print(f"[error] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if args.percentages:
        cfg["percentages_list"] = [int(x) for x in args.percentages.split(",")]
    cfg["new"] = args.new

    print(f"[config] Loaded {config_path}")
    run_experiment(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
