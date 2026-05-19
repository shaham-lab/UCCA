"""
Total Correlation Experiment
============================
Trains a model on unpaired multi-view data and evaluates the total
correlation on a held-out test set.  Repeats the experiment and
reports mean +/- std.

Usage:
    python exps/exp_total_correlation.py --method sca --data handwritten
"""

import sys
import os
import argparse

import yaml
import numpy as np
import torch

# Ensure project root is on the path so imports work from any cwd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import get_data, prepare_data
from utils import (
    get_total_correlation,
    get_cross_view_accuracy,
    save_results
)


from exps.methods import run_method_dispatch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict, preloaded_data=None, config_path=None, save=True):
    """Run the full experiment loop described by *cfg*.

    When output_size is a list, a single JSON is saved with a 'dims' key
    mapping each dim to its {correlations, mean, std}.  When output_size is
    a scalar the JSON has the flat legacy structure.
    """
    method = cfg["method"]
    data_kind = cfg["data"]
    n_repeats = cfg.get("n_repeats", 5)
    n_test = cfg.get("n_test", 256)
    seed = cfg.get("seed", 42)
    results_dir = cfg.get("results_dir", "results")
    output_size = cfg.get("output_size", 2)
    multi_dim = isinstance(output_size, list)

    if preloaded_data is not None:
        X_train, X_test, Y_train, Y_test, L_X_train, L_Y_train, L_test = preloaded_data
    else:
        np.random.seed(seed)
        torch.manual_seed(seed)

        A, B, labels = get_data(
            data_kind,
            n_points=cfg.get("n_points"),
            view1_idx=cfg.get("view1_idx"),
            view2_idx=cfg.get("view2_idx"),
        )
        X_train, X_test, Y_train, Y_test, L_X_train, L_Y_train, L_test = prepare_data(A, B, n_test,
                                                         split=cfg.get("split", "unpaired"),
                                                         labels=labels)

    print(f"[exp] Train: {X_train.shape}, Test: {X_test.shape}")

    import time
    # ------------------------------------------------------------------
    # Run repeats
    # ------------------------------------------------------------------
    # Metrics to track: tc, runtime
    metrics_keys = ["tc", "runtime"]
    if L_test is not None:
        metrics_keys += ["acc_v1_to_v2_linear", "acc_v2_to_v1_linear",
                        "acc_v1_to_v2_knn", "acc_v2_to_v1_knn"]

    if multi_dim:
        dim_metrics = {d: {k: [] for k in metrics_keys} for d in output_size}
    else:
        scalar_metrics = {k: [] for k in metrics_keys}

    # Megatest trick to get both train and test projections
    X_mega = torch.cat([X_train, X_test], dim=0)
    Y_mega = torch.cat([Y_train, Y_test], dim=0)
    n_train = len(X_train)

    for i in range(n_repeats):
        start_time = time.time()
        # Use mega set to get train and test projections at once
        result = run_method_dispatch(method, X_train, Y_train, X_mega, Y_mega, cfg)
        runtime = time.time() - start_time

        if multi_dim:
            for d, (o1, o2) in result.items():
                out1_tr, out1_te = o1[:n_train], o1[n_train:]
                out2_tr, out2_te = o2[:n_train], o2[n_train:]
                
                tc = get_total_correlation(out1_te, out2_te)
                
                dim_metrics[d]["tc"].append(tc)
                dim_metrics[d]["runtime"].append(runtime)

                if L_test is not None:
                    v12l = get_cross_view_accuracy(out1_tr, L_X_train, out2_te, L_test, 'linear')
                    v21l = get_cross_view_accuracy(out2_tr, L_Y_train, out1_te, L_test, 'linear')
                    v12k = get_cross_view_accuracy(out1_tr, L_X_train, out2_te, L_test, 'knn')
                    v21k = get_cross_view_accuracy(out2_tr, L_Y_train, out1_te, L_test, 'knn')
                    dim_metrics[d]["acc_v1_to_v2_linear"].append(v12l)
                    dim_metrics[d]["acc_v2_to_v1_linear"].append(v21l)
                    dim_metrics[d]["acc_v1_to_v2_knn"].append(v12k)
                    dim_metrics[d]["acc_v2_to_v1_knn"].append(v21k)

                print(f"  repeat {i + 1}/{n_repeats}  dim={d}: tc={tc:.4f}")
                
                if i == n_repeats - 1:
                    dim_metrics[d]["outputs"] = (out1_te.tolist(), out2_te.tolist())
        else:
            o1, o2 = result
            out1_tr, out1_te = o1[:n_train], o1[n_train:]
            out2_tr, out2_te = o2[:n_train], o2[n_train:]

            tc = get_total_correlation(out1_te, out2_te)
            
            scalar_metrics["tc"].append(tc)
            scalar_metrics["runtime"].append(runtime)

            if L_test is not None:
                v12l = get_cross_view_accuracy(out1_tr, L_X_train, out2_te, L_test, 'linear')
                v21l = get_cross_view_accuracy(out2_tr, L_Y_train, out1_te, L_test, 'linear')
                v12k = get_cross_view_accuracy(out1_tr, L_X_train, out2_te, L_test, 'knn')
                v21k = get_cross_view_accuracy(out2_tr, L_Y_train, out1_te, L_test, 'knn')
                scalar_metrics["acc_v1_to_v2_linear"].append(v12l)
                scalar_metrics["acc_v2_to_v1_linear"].append(v21l)
                scalar_metrics["acc_v1_to_v2_knn"].append(v12k)
                scalar_metrics["acc_v2_to_v1_knn"].append(v21k)

            print(f"  repeat {i + 1}/{n_repeats}: tc={tc:.4f}, time={runtime:.2f}s")

            if i == n_repeats - 1:
                scalar_metrics["outputs"] = (out1_te.tolist(), out2_te.tolist())

    # ------------------------------------------------------------------
    # Aggregate and print summary
    # ------------------------------------------------------------------
    exp_name = cfg.get("exp_name", None)
    method_name = cfg.get("method_name", method)

    if multi_dim:
        dims_summary = {}
        for d in output_size:
            summary = {}
            for k in metrics_keys:
                vals = dim_metrics[d][k]
                summary[k] = vals
                m_key = "mean" if k == "tc" else f"{k}_mean"
                s_key = "std" if k == "tc" else f"{k}_std"
                summary[m_key] = float(np.mean(vals))
                summary[s_key] = float(np.std(vals))
            dims_summary[d] = summary
            if "outputs" in dim_metrics[d]:
                dims_summary[d]["outputs"] = dim_metrics[d]["outputs"]
            mu, std = summary["mean"], summary["std"]
            print(f"\n[result] {method} / {data_kind} / dim={d}: tc = {mu:.4f} +/- {std:.4f}")
            if "acc_v1_to_v2_linear_mean" in summary:
                v12l, v12ls = summary["acc_v1_to_v2_linear_mean"], summary["acc_v1_to_v2_linear_std"]
                v21l, v21ls = summary["acc_v2_to_v1_linear_mean"], summary["acc_v2_to_v1_linear_std"]
                print(f"  Accuracy (linear): V1 -> V2: {v12l:.4f} +/- {v12ls:.4f}, V2 -> V1: {v21l:.4f} +/- {v21ls:.4f}")
            if "acc_v1_to_v2_knn_mean" in summary:
                v12k, v12ks = summary["acc_v1_to_v2_knn_mean"], summary["acc_v1_to_v2_knn_std"]
                v21k, v21ks = summary["acc_v2_to_v1_knn_mean"], summary["acc_v2_to_v1_knn_std"]
                print(f"  Accuracy (knn):    V1 -> V2: {v12k:.4f} +/- {v12ks:.4f}, V2 -> V1: {v21k:.4f} +/- {v21ks:.4f}")

        results = {
            "method": method,
            "data": data_kind,
            "method_name": method_name,
            "config": cfg,
            "dims": {str(d): v for d, v in dims_summary.items()},
        }
    else:
        summary = {}
        for k in metrics_keys:
            vals = scalar_metrics[k]
            summary[k] = vals
            m_key = "mean" if k == "tc" else f"{k}_mean"
            s_key = "std" if k == "tc" else f"{k}_std"
            summary[m_key] = float(np.mean(vals))
            summary[s_key] = float(np.std(vals))
        
        mu, std = summary["mean"], summary["std"]
        print(f"\n[result] {method} / {data_kind}: tc = {mu:.4f} +/- {std:.4f}")
        if "acc_v1_to_v2_linear_mean" in summary:
            v12l, v12ls = summary["acc_v1_to_v2_linear_mean"], summary["acc_v1_to_v2_linear_std"]
            v21l, v21ls = summary["acc_v2_to_v1_linear_mean"], summary["acc_v2_to_v1_linear_std"]
            print(f"  Accuracy (linear): V1 -> V2: {v12l:.4f} +/- {v12ls:.4f}, V2 -> V1: {v21l:.4f} +/- {v21ls:.4f}")
        if "acc_v1_to_v2_knn_mean" in summary:
            v12k, v12ks = summary["acc_v1_to_v2_knn_mean"], summary["acc_v1_to_v2_knn_std"]
            v21k, v21ks = summary["acc_v2_to_v1_knn_mean"], summary["acc_v2_to_v1_knn_std"]
            print(f"  Accuracy (knn):    V1 -> V2: {v12k:.4f} +/- {v12ks:.4f}, V2 -> V1: {v21k:.4f} +/- {v21ks:.4f}")
        results = {
            "method": method,
            "data": data_kind,
            "method_name": method_name,
            "config": cfg,
            **summary
        }
        if "outputs" in scalar_metrics:
            results["outputs"] = scalar_metrics["outputs"]

    if save:
        if exp_name is None:
            if config_path:
                exp_name = os.path.splitext(os.path.basename(config_path))[0]
            else:
                exp_name = f"{method}_{data_kind}"
        save_results(exp_name, results, results_dir)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Total Correlation Experiment")
    parser.add_argument("--method", type=str, default="ucca",
                        help="Method name (e.g. 'ucca')")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset name (e.g. 'handwritten')")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: configs/{method}_{data}.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve config path
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

    print(f"[config] Loaded {config_path}")
    print(f"[config] {cfg}")

    run_experiment(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
