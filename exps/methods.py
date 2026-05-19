"""
Shared method runners for experiments.
"""
import torch

def run_method_dispatch(method: str, X_train, Y_train, X_test, Y_test, cfg: dict):
    """
    Dispatch to the appropriate model runner.

    Returns:
        (output1, output2) tuple of tensors, OR
        dict[int -> (output1, output2)] if output_size is a list.
    """
    # Keys consumed by the experiment framework (not forwarded to the model)
    _experiment_keys = {"method", "method_name", "data", "n_test", "n_repeats", "seed",
                        "results_dir", "n_points", "view1_idx", "view2_idx", "split", "exp_name"}

    def _to_tensors(out):
        """Convert a (out1, out2) tuple or dict of tuples to torch tensors."""
        if isinstance(out, dict):
            return {d: (torch.tensor(o1, dtype=torch.float32),
                        torch.tensor(o2, dtype=torch.float32))
                    for d, (o1, o2) in out.items()}
        out1, out2 = out
        return torch.tensor(out1, dtype=torch.float32), torch.tensor(out2, dtype=torch.float32)

    if method == "sca":
        from baselines.sca.run import run_usca
        model_kwargs = {k: v for k, v in cfg.items() if k not in _experiment_keys}
        return _to_tensors(run_usca(
            X_train, Y_train,
            output_size=model_kwargs.pop("output_size", 10),
            X_test=X_test, Y_test=Y_test,
            **model_kwargs,
        ))
    elif method == "cca":
        from baselines.cca.run import run_cca
        model_kwargs = {k: v for k, v in cfg.items() if k not in _experiment_keys}
        return _to_tensors(run_cca(
            X_train, Y_train,
            output_size=model_kwargs.pop("output_size", 2),
            X_test=X_test, Y_test=Y_test,
            **model_kwargs,
        ))
    elif method == "pca_cca":
        from baselines.pca_cca.run import run_pca_cca
        model_kwargs = {k: v for k, v in cfg.items() if k not in _experiment_keys}
        return _to_tensors(run_pca_cca(
            X_train, Y_train,
            output_size=model_kwargs.pop("output_size", 2),
            X_test=X_test, Y_test=Y_test,
            **model_kwargs,
        ))
    elif method == "ucca":
        from ucca.run import run_qap_cca
        model_kwargs = {k: v for k, v in cfg.items() if k not in _experiment_keys}
        return _to_tensors(run_qap_cca(
            X_train, Y_train,
            output_size=model_kwargs.pop("output_size", 2),
            X_test=X_test, Y_test=Y_test,
            **model_kwargs,
        ))
    elif method == "jmds_cca":
        from baselines.jmds_cca.run import run_jmds_cca
        model_kwargs = {k: v for k, v in cfg.items() if k not in _experiment_keys}
        return _to_tensors(run_jmds_cca(
            X_train, Y_train,
            output_size=model_kwargs.pop("output_size", 2),
            X_test=X_test, Y_test=Y_test,
            **model_kwargs,
        ))
    elif method == "scot1_cca":
        from baselines.scot1_cca.run import run_scot1_cca
        model_kwargs = {k: v for k, v in cfg.items() if k not in _experiment_keys}
        return _to_tensors(run_scot1_cca(
            X_train, Y_train,
            output_size=model_kwargs.pop("output_size", 2),
            X_test=X_test, Y_test=Y_test,
            **model_kwargs,
        ))
    elif method == "scot2_cca":
        from baselines.scot2_cca.run import run_scot2_cca
        model_kwargs = {k: v for k, v in cfg.items() if k not in _experiment_keys}
        return _to_tensors(run_scot2_cca(
            X_train, Y_train,
            output_size=model_kwargs.pop("output_size", 2),
            X_test=X_test, Y_test=Y_test,
            **model_kwargs,
        ))
    elif method == "uca18":
        from baselines.uca18.run import run_uca18
        model_kwargs = {k: v for k, v in cfg.items() if k not in _experiment_keys}
        return _to_tensors(run_uca18(
            X_train, Y_train,
            output_size=model_kwargs.pop("output_size", 2),
            X_test=X_test, Y_test=Y_test,
            **model_kwargs,
        ))
    elif method == "uca18_linear":
        from baselines.uca18_linear.run import run_uca18_linear
        model_kwargs = {k: v for k, v in cfg.items() if k not in _experiment_keys}
        return _to_tensors(run_uca18_linear(
            X_train, Y_train,
            output_size=model_kwargs.pop("output_size", 2),
            X_test=X_test, Y_test=Y_test,
            **model_kwargs,
        ))
    else:
        raise ValueError(f"Unknown method: {method}")
