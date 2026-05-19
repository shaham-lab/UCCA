"""
N-Balance Experiment
====================
Varies the number of samples in one view as a percentage of the total samples
(balance) while keeping the other view at 100%.

The percentage p represents: n_subset / (n_subset + n_full) = p / 100
where n_full is the total number of training samples in the constant view.

Usage:
    python exps/exp_n_balance.py --method ucca --data handwritten --subset_view 1
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
    """Run the N-balance experiment."""
    method = cfg["method"]
    data_kind = cfg["data"]
    n_repeats = cfg.get("n_repeats", 5)
    n_test = cfg.get("n_test", 256)
    seed = cfg.get("seed", 42)
    subset_view = cfg.get("subset_view", 1)
    results_dir = "results/n_balance"
    output_size = cfg.get("output_size", 2)
    
    # We only handle scalar output_size for now in this exp
    if isinstance(output_size, list):
        print("[warn] exp_n_balance only supports scalar output_size. Using first element.")
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
    
    # Add view suffix to avoid overwriting
    exp_name = f"{exp_name}_v{subset_view}"

    # Check for existing results (checkpointing)
    results_path = os.path.join(results_dir, f"{exp_name}.json")
    
    default_percentages = [30, 40]
    
    if not cfg.get("new", False) and os.path.exists(results_path):
        print(f"[checkpoint] Loading existing results from {results_path}")
        with open(results_path, "r") as f:
            final_results = json.load(f)
        results_by_pct = final_results.get("results", {})
        percentages = final_results.get("percentages", default_percentages)
        
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
            percentages = default_percentages
        else:
            percentages = cfg["percentages_list"]
            
        final_results = {
            "method": method,
            "data": data_kind,
            "subset_view": subset_view,
            "method_name": cfg.get("method_name", method),
            "config": cfg,
            "percentages": percentages,
            "results": results_by_pct
        }

    print(f"[exp] Starting/Resuming N-balance sweep for {method} on {data_kind} (subset_view={subset_view})")

    def run_single_pct(pct):
        if pct >= 100:
            print(f"[skip] pct={pct} is too high for balance definition. Max is < 100 (practical max 50).")
            return
        
        # n_subset / (n_subset + n_full) = p / 100
        # n_subset = (p * n_full) / (100 - p)
        n_subset = int(n_train_full * pct / (100 - pct))
        n_subset = min(n_subset, n_train_full) # Cannot exceed full data pool
        
        print(f"\n[exp] pct={pct}% (n_subset={n_subset}, n_full={n_train_full})")
        
        correlations = []
        for i in range(n_repeats):
            if subset_view == 1:
                idx = torch.randperm(n_train_full)[:n_subset]
                X_train = X_train_full[idx]
                Y_train = Y_train_full
            else:
                idx = torch.randperm(n_train_full)[:n_subset]
                X_train = X_train_full
                Y_train = Y_train_full[idx]
            
            # Update config for this run, ensuring custom sweep args aren't passed
            point_cfg = cfg.copy()
            point_cfg.pop("percentages_list", None)
            point_cfg.pop("new", None)
            point_cfg.pop("subset_view", None)

            result = run_method_dispatch(method, X_train, Y_train, X_test, Y_test, point_cfg)
            out1, out2 = result
            tc = get_total_correlation(out1, out2)
            correlations.append(tc)
            
        mu = float(np.mean(correlations))
        std = float(np.std(correlations))
        
        result_entry = {
            "correlations": correlations,
            "mean": mu,
            "std": std,
            "n_subset": n_subset,
            "n_full": n_train_full
        }
        
        # Reload just before saving to support parallel runs
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    latest_results = json.load(f)
                latest_results["results"][str(pct)] = result_entry
                # Ensure percentages list is also updated if we are adding a new one
                if pct not in latest_results["percentages"]:
                    latest_results["percentages"].append(pct)
                    latest_results["percentages"].sort()
                final_results.update(latest_results)
            except Exception as e:
                print(f"[warn] Failed to reload/merge results: {e}")
                final_results["results"][str(pct)] = result_entry
        else:
            final_results["results"][str(pct)] = result_entry
            
        save_results(exp_name, final_results, results_dir)
        print(f"[result] pct={pct}%: tc = {mu:.4f} +/- {std:.4f}")
        
    to_run = [pct for pct in percentages if str(pct) not in results_by_pct]
    
    if to_run:
        for pct in to_run:
            run_single_pct(pct)

    return final_results


def main():
    parser = argparse.ArgumentParser(description="N-Balance Experiment")
    parser.add_argument("--method", type=str, default="ucca")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--percentages", type=str, default=None, help="Comma-separated list of percentages (e.g. 10,40,50)")
    parser.add_argument("--subset_view", type=int, default=1, choices=[1, 2], help="Which view to subset")
    parser.add_argument("--new", action="store_true", help="Start new experiment")
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default",
            f"{args.method}_{args.data}.yaml"
        )

    if not os.path.exists(config_path):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs",
            f"{args.method}_{args.data}.yaml"
        )

    if not os.path.exists(config_path):
        print(f"[error] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if args.percentages:
        cfg["percentages_list"] = sorted([int(x) for x in args.percentages.split(",")])
    cfg["new"] = args.new
    cfg["subset_view"] = args.subset_view

    print(f"[config] Loaded {config_path}")
    run_experiment(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
