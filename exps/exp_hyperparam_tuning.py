"""
Hyperparameter Tuning Experiment
================================
Grid-searches over list-valued config parameters, running the total
correlation experiment for each combination.

Config values that are lists are treated as tunable; scalar values are
fixed.  For example:

    output_size: [5, 10, 20]   # tuned over 3 values
    loss_func: CE              # fixed

Usage:
    python exps/exp_hyperparam_tuning.py --method sca --data handwritten
"""

import sys
import os
import argparse
import itertools

import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_data, prepare_data
from exp_total_correlation import run_experiment
from utils import save_results


# ---------------------------------------------------------------------------
# Grid helper
# ---------------------------------------------------------------------------

def _build_grid(cfg: dict):
    """Separate list-valued (tunable) from scalar (fixed) keys."""
    fixed = {}
    tunable_keys = []
    tunable_values = []

    for k, v in cfg.items():
        if isinstance(v, list):
            tunable_keys.append(k)
            tunable_values.append(v)
        else:
            fixed[k] = v

    combos = []
    if not tunable_keys:
        combos.append(fixed)
    else:
        for values in itertools.product(*tunable_values):
            point = fixed.copy()
            for k, v in zip(tunable_keys, values):
                point[k] = v
            combos.append(point)

    return tunable_keys, combos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tuning(cfg: dict, config_path=None):
    method = cfg.get("method", "unknown")
    data_kind = cfg.get("data", "unknown")
    results_dir = cfg.get("results_dir", "results")
    seed = cfg.get("seed", 42)
    n_test = cfg.get("n_test", 256)

    tunable_keys, combos = _build_grid(cfg)
    n_total = len(combos)

    # Load data once (seeded for reproducible split)
    np.random.seed(seed)
    torch.manual_seed(seed)
    embed_A, embed_B, labels = get_data(
        data_kind,
        n_points=cfg.get("n_points"),
        view1_idx=cfg.get("view1_idx"),
        view2_idx=cfg.get("view2_idx"),
    )
    preloaded_data = prepare_data(embed_A, embed_B, n_test,
                                   split=cfg.get("split", "unpaired"))

    print(f"[tuning] {n_total} configuration(s) to evaluate\n")

    all_results = []
    for idx, point_cfg in enumerate(combos, 1):
        hp_str = ", ".join(f"{k}={point_cfg[k]}" for k in tunable_keys)
        print(f"{'='*60}")
        print(f"[tuning] Config {idx}/{n_total}: {hp_str}")
        print(f"{'='*60}")

        result = run_experiment(point_cfg, preloaded_data=preloaded_data, save=False)
        result["tuned_params"] = {k: point_cfg[k] for k in tunable_keys}
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("[tuning] Summary (best -> worst)")
    print(f"{'='*60}")
    for r in sorted(all_results, key=lambda r: r["mean"], reverse=True):
        hp = "  ".join(f"{k}={r['tuned_params'][k]}" for k in tunable_keys)
        print(f"  {hp:40s}  mean={r['mean']:.4f}  std={r['std']:.4f}")

    if config_path:
        exp_name = os.path.splitext(os.path.basename(config_path))[0] + "_tuning"
    else:
        exp_name = f"{method}_{data_kind}_tuning"
    save_results(exp_name, all_results, results_dir)
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument("--method", type=str, default="ucca")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or os.path.join(
        os.path.dirname(__file__), "..", "configs", "tuning",
        f"{args.method}_{args.data}.yaml"
    )

    if not os.path.exists(config_path):
        print(f"[error] Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"[config] Loaded {config_path}")
    run_tuning(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
