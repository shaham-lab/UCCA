"""
Anchor Kind Experiment
=======================
Evaluates the performance of ucca by varying the kind of anchors (qap_anchor_kind).
This experiment is specifically for the ucca method.

Usage:
    python exps/exp_anchors_kind.py --data handwritten
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


def run_anchors_kind(cfg: dict, config_path=None, restart=False):
    """Run the anchors kind experiment."""
    method = cfg.get("method", "ucca")
    if method != "ucca":
        print(f"[warning] method is {method}, but this experiment is designed for ucca.")
    
    data_kind = cfg["data"]
    results_dir = "results/anchors_kind"
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
        results_by_kind = final_results.get("results", {})
        anchor_kind_list = final_results.get("anchor_kinds", ["kmeans", "random", "paired_kmeans", "paired_kmeans_reversed"])
    else:
        if restart:
            print("[checkpoint] Starting new experiment (--new flag provided)")
        results_by_kind = {}
        anchor_kind_list = ["kmeans", "random", "paired_kmeans", "paired_kmeans_reversed"]
        final_results = {
            "method": method,
            "data": data_kind,
            "method_name": cfg.get("method_name", method),
            "config": cfg,
            "anchor_kinds": anchor_kind_list,
            "results": results_by_kind
        }

    # Preload data to be consistent across kinds
    np.random.seed(seed)
    torch.manual_seed(seed)

    embed_A, embed_B, labels = get_data(
        data_kind,
        n_points=cfg.get("n_points"),
        view1_idx=cfg.get("view1_idx"),
        view2_idx=cfg.get("view2_idx"),
    )
    # Use unpaired split as in total_correlation by default
    default_preloaded_data = prepare_data(embed_A, embed_B, n_test,
                                           split=cfg.get("split", "unpaired"))
    
    # Paired split for the upper bound and paired kmemoids
    paired_preloaded_data = prepare_data(embed_A, embed_B, n_test,
                                          split="paired")

    print(f"[exp] Starting/Resuming Anchor Kind sweep for {method} on {data_kind}")
    
    for kind in anchor_kind_list:
        if kind in results_by_kind:
            print(f"[checkpoint] Skipping anchor_kind = {kind} (already computed)")
            continue

        print(f"\n{'='*60}")
        print(f"[exp] anchor_kind = {kind}")
        print(f"{'='*60}")
        
        # Update config for this point
        point_cfg = cfg.copy()
        
        if kind == "paired_cca":
            point_cfg["method"] = "cca"
            data = paired_preloaded_data
        elif kind in ["paired_kmemoids", "paired_kmeans", "paired_kmeans_reversed"]:
            point_cfg[f"{method}_anchor_kind" if f"{method}_anchor_kind" in point_cfg else "qap_anchor_kind"] = kind
            data = paired_preloaded_data
        else:
            point_cfg[f"{method}_anchor_kind" if f"{method}_anchor_kind" in point_cfg else "qap_anchor_kind"] = kind
            data = default_preloaded_data
        
        # Run experiment (don't save individual results)
        res = run_experiment(point_cfg, preloaded_data=data, save=False)
        
        # Handle multi-dim vs scalar results from exp_total_correlation
        if "dims" in res:
            # If multi-dim, we pick the first one for the summary plot or similar
            first_dim = list(res["dims"].keys())[0]
            summary = res["dims"][first_dim]
        else:
            summary = res
            
        results_by_kind[kind] = {
            "mean": summary["mean"],
            "std": summary["std"],
            "correlations": summary["tc"]
        }
        print(f"[result] anchor_kind={kind}: tc = {summary['mean']:.4f} +/- {summary['std']:.4f}")

        # Save checkpoint after each step
        final_results["results"] = results_by_kind
        save_results(exp_name, final_results, results_dir)

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Anchor Kind Experiment for QAP-CCA")
    parser.add_argument("--data", type=str, required=True, help="Dataset name")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--n_points", type=int, default=None)
    parser.add_argument("--n_repeats", type=int, default=None)
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

    print(f"[config] Loaded {config_path}")
    run_anchors_kind(cfg, config_path=config_path, restart=args.new)


if __name__ == "__main__":
    main()
