"""
N-Anchors Experiment
====================
Evaluates the performance of ucca by varying the number of anchors (n_anchors).
This experiment is specifically for the ucca method.

Usage:
    python exps/exp_n_anchors.py --data handwritten
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
from exp_total_correlation import run_experiment
from utils import save_results



def run_n_anchors(cfg: dict, config_path=None, restart=False):
    """Run the N-anchors experiment."""
    method = cfg.get("method", "ucca")
    if method != "ucca":
        print(f"[warning] method is {method}, but this experiment is designed for ucca.")
    
    data_kind = cfg["data"]
    results_dir = "results/n_anchors"
    seed = cfg.get("seed", 42)
    n_test = cfg.get("n_test", 256)

    if config_path:
        exp_name = os.path.splitext(os.path.basename(config_path))[0]
    else:
        exp_name = f"{method}_{data_kind}"

    # Check for existing results (checkpointing)
    results_path = os.path.join(results_dir, f"{exp_name}.json")
    if not restart and os.path.exists(results_path):
        print(f"[checkpoint] Loading existing results from {results_path}")
        with open(results_path, "r") as f:
            final_results = json.load(f)
        results_by_n = final_results.get("results", {})
        # Ensure we use the original n_anchors_list if possible, or reconstruct it
        n_anchors_list = final_results.get("n_anchors", [15, 16, 17, 18, 19, 20])
        
        # Merge with provided n_anchors if any
        if cfg.get("n_anchors_list"):
            for n in cfg["n_anchors_list"]:
                if n not in n_anchors_list:
                    n_anchors_list.append(n)
            n_anchors_list.sort()
            final_results["n_anchors"] = n_anchors_list
    else:
        if restart:
            print("[checkpoint] Starting new experiment (--new flag provided)")
        results_by_n = {}
        # Default n_anchors_list if not provided and no checkpoint
        if not cfg.get("n_anchors_list"):
            n_anchors_list = [15, 16, 17, 18, 19, 20]
        else:
            n_anchors_list = cfg["n_anchors_list"]

        final_results = {
            "method": method,
            "data": data_kind,
            "method_name": cfg.get("method_name", method),
            "config": cfg,
            "n_anchors": n_anchors_list,
            "results": results_by_n
        }

    # Preload data to be consistent across n_anchors values
    np.random.seed(seed)
    torch.manual_seed(seed)

    embed_A, embed_B, labels = get_data(
        data_kind,
        n_points=cfg.get("n_points"),
        view1_idx=cfg.get("view1_idx"),
        view2_idx=cfg.get("view2_idx"),
    )
    # Use unpaired split as in total_correlation by default
    preloaded_data = prepare_data(embed_A, embed_B, n_test,
                                   split=cfg.get("split", "unpaired"))

    print(f"[exp] Starting/Resuming N-anchors sweep for {method} on {data_kind}")

    def run_single_n(n):
        print(f"\n{'='*60}\n[exp] n_anchors = {n}\n{'='*60}")
        
        # Update config for this point
        point_cfg = cfg.copy()
        point_cfg["n_anchors"] = n
        point_cfg.pop("n_anchors_list", None)
        point_cfg.pop("n_jobs", None)
        
        # Run experiment (don't save individual results)
        res = run_experiment(point_cfg, preloaded_data=preloaded_data, save=False)
        
        # Handle multi-dim vs scalar results from exp_total_correlation
        if "dims" in res:
            first_dim = list(res["dims"].keys())[0]
            summary = res["dims"][first_dim]
        else:
            summary = res
            
        result = {
            "mean": summary["mean"],
            "std": summary["std"],
            "correlations": summary["tc"]
        }
        
        results_by_n[str(n)] = result
        final_results["results"] = results_by_n
        save_results(exp_name, final_results, results_dir)
        print(f"[result] n_anchors={n}: tc = {summary['mean']:.4f} +/- {summary['std']:.4f}")
        
        return n, result

    # Identify which n need to be run
    to_run = [n for n in n_anchors_list if str(n) not in results_by_n]
    
    if to_run:
        for n in to_run:
            run_single_n(n)

    return final_results


def main():
    parser = argparse.ArgumentParser(description="N-Anchors Experiment for QAP-CCA")
    parser.add_argument("--data", type=str, required=True, help="Dataset name")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--n_points", type=int, default=None)
    parser.add_argument("--n_repeats", type=int, default=None)
    parser.add_argument("--n_anchors", type=str, default=None, help="Comma-separated list of anchor counts")
    parser.add_argument("--new", action="store_true", help="Start a new experiment, ignoring checkpoints")
    
    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default",
            f"ucca_{args.data}.yaml"
        )

    if not os.path.exists(config_path):
        print(f"[error] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Override with CLI args if provided
    if args.n_points: cfg["n_points"] = args.n_points
    if args.n_repeats: cfg["n_repeats"] = args.n_repeats
    if args.n_anchors:
        cfg["n_anchors_list"] = [int(x) for x in args.n_anchors.split(",")]

    print(f"[config] Loaded {config_path}")
    run_n_anchors(cfg, config_path=config_path, restart=args.new)


if __name__ == "__main__":
    main()
