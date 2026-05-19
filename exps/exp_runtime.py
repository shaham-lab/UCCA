"""
Runtime vs. Total Correlation Experiment
========================================
Evaluates the trade-off between runtime and performance by varying 
the number of QAP runs or anchor alignment runs for QAP-CCA.

Usage:
    python exps/exp_runtime.py --data handwritten --param anchors_n_runs --values 1 5 10 20 30 50 100 200 500
"""

import sys
import os
import argparse
import yaml
import json
import numpy as np
import torch

# Ensure project root is on the path so imports work from any cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_data, prepare_data
from exp_total_correlation import run_experiment
from utils import save_results


def run_runtime_exp(cfg: dict, param_to_sweep: str, values: list, config_path=None, restart=False):
    """Run the runtime experiment sweep."""
    method = cfg.get("method", "ucca")
    if method != "ucca":
        print(f"[warning] method is {method}, but this experiment is designed for ucca.")
    
    data_kind = cfg["data"]
    results_dir = "results/runtime"
    os.makedirs(results_dir, exist_ok=True)
    seed = cfg.get("seed", 42)
    n_test = cfg.get("n_test", 256)

    if config_path:
        base_name = os.path.splitext(os.path.basename(config_path))[0]
        # Avoid double param name if already in config name
        if param_to_sweep in base_name:
            exp_name = base_name
        else:
            exp_name = f"{base_name}_{param_to_sweep}"
    else:
        exp_name = f"{method}_{data_kind}_{param_to_sweep}"

    # Check for existing results (checkpointing)
    results_path = os.path.join(results_dir, f"{exp_name}.json")
    if not restart and os.path.exists(results_path):
        print(f"[checkpoint] Loading existing results from {results_path}")
        with open(results_path, "r") as f:
            final_results = json.load(f)
        results_by_val = final_results.get("results", {})
        sweep_values = final_results.get("sweep_values", values)
    else:
        if restart:
            print("[checkpoint] Starting new experiment (--new flag provided)")
        results_by_val = {}
        sweep_values = values
        final_results = {
            "method": method,
            "data": data_kind,
            "param_swept": param_to_sweep,
            "method_name": cfg.get("method_name", method),
            "config": cfg,
            "sweep_values": sweep_values,
            "results": results_by_val
        }

    # Preload data to be consistent
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

    print(f"[exp] Starting/Resuming Runtime sweep ({param_to_sweep}) for {method} on {data_kind}")
    
    for val in sweep_values:
        if str(val) in results_by_val:
            print(f"[checkpoint] Skipping {param_to_sweep} = {val} (already computed)")
            continue

        print(f"\n{'='*60}")
        print(f"[exp] {param_to_sweep} = {val}")
        print(f"{'='*60}")
        
        # Update config for this point
        point_cfg = cfg.copy()
        point_cfg[param_to_sweep] = val
        
        # Run experiment (don't save individual results)
        res = run_experiment(point_cfg, preloaded_data=preloaded_data, save=False)
        
        # Handle multi-dim vs scalar results from exp_total_correlation
        if "dims" in res:
            # If multi-dim, we pick the first one (or the only one)
            first_dim = list(res["dims"].keys())[0]
            summary = res["dims"][first_dim]
        else:
            summary = res
            
        results_by_val[str(val)] = {
            "tc_mean": summary["mean"],
            "tc_std": summary["std"],
            "correlations": summary["tc"],
            "runtime_mean": summary["runtime_mean"],
            "runtime_std": summary["runtime_std"],
            "runtimes": summary["runtime"]
        }
        print(f"[result] {param_to_sweep}={val}: tc = {summary['mean']:.4f}, time = {summary['runtime_mean']:.2f}s")

        # Save checkpoint after each step
        final_results["results"] = results_by_val
        save_results(exp_name, final_results, results_dir)

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Runtime Experiment for QAP-CCA")
    parser.add_argument("--data", type=str, required=True, help="Dataset name")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--qap", action="store_true", help="Sweep qap_n_runs instead of anchors_n_runs")
    parser.add_argument("--values", type=int, nargs="+", default=[1, 5, 10, 20, 30, 50, 100, 200, 500])
    parser.add_argument("--n_points", type=int, default=None)
    parser.add_argument("--n_repeats", type=int, default=None)
    parser.add_argument("--new", action="store_true", help="Start a new experiment, ignoring checkpoints")
    
    args = parser.parse_args()

    param_to_sweep = "qap_n_runs" if args.qap else "anchors_n_runs"

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
    run_runtime_exp(cfg, param_to_sweep, args.values, config_path=config_path, restart=args.new)


if __name__ == "__main__":
    main()
