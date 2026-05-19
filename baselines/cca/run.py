import numpy as np
from sklearn.cross_decomposition import CCA


def run_cca(X_train, Y_train, output_size, X_test=None, Y_test=None, **kwargs):
    """
    Run sklearn CCA.

    Args:
        X_train, Y_train: Training tensors.
        output_size (int or list[int]): Number of components. If a list,
            CCA is fit once with max(output_size) and results are returned
            for each dim as a dict {dim: (out1, out2)}.
        X_test, Y_test: Test tensors (default to train if None).

    Returns:
        (output1, output2) if output_size is int, else dict[int -> (out1, out2)].
    """
    if X_test is None or Y_test is None:
        X_test = X_train
        Y_test = Y_train

    dims = output_size if isinstance(output_size, list) else None
    fit_size = max(dims) if dims is not None else output_size

    cca = CCA(n_components=fit_size)
    cca.fit(X_train.cpu().numpy(), Y_train.cpu().numpy())

    out1_train, out2_train = cca.transform(X_train.cpu().numpy(), Y_train.cpu().numpy())
    out1, out2 = cca.transform(X_test.cpu().numpy(), Y_test.cpu().numpy())

    # Normalize each view separately (zero mean, unit variance)
    X_mean = out1_train.mean(axis=0)
    Y_mean = out2_train.mean(axis=0)
    X_std = out1_train.std(axis=0)
    Y_std = out2_train.std(axis=0)
    out1 = (out1 - X_mean) / (X_std + 1e-8)
    out2 = (out2 - Y_mean) / (Y_std + 1e-8)

    if dims is None:
        return out1, out2

    return {d: (out1_full[:, :d], out2_full[:, :d]) for d in dims}
