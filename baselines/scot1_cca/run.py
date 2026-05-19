"""
SCOT v1 runner.
"""

import torch
import numpy as np
from sklearn.cross_decomposition import CCA

from baselines.scot.src.scotv1 import SCOT

def run_scot1_cca(X_train, Y_train, output_size, X_test=None, Y_test=None, k=20, eps=0.01, **kwargs):
    if X_test is None or Y_test is None:
        X_test = X_train
        Y_test = Y_train

    subsample_size = 2000
    if X_train.shape[0] > subsample_size:
        subsample_indices = np.random.choice(X_train.shape[0], subsample_size, replace=False)
        X_train = X_train[subsample_indices]
        Y_train = Y_train[subsample_indices]

    dims = output_size if isinstance(output_size, list) else None
    fit_size = max(dims) if dims is not None else output_size

    # Run SCOT1
    X_train_np = X_train.cpu().numpy()
    Y_train_np = Y_train.cpu().numpy()
    scot = SCOT(X_train_np, Y_train_np)
    scot.align(k=k, e=eps, **kwargs)

    # Get coupling and compute pseudo-paired Y matching X
    P = scot.coupling  # Shape (N, N) relating X_i and Y_j
    weights = np.sum(P, axis=1, keepdims=True)
    Y_train_paired = (P @ Y_train_np) / (weights + 1e-10)

    cca = CCA(n_components=fit_size)
    cca.fit(X_train_np, Y_train_paired)
    out1_full, out2_full = cca.transform(X_test.cpu().numpy(), Y_test.cpu().numpy())


    if dims is None:
        return out1_full, out2_full

    return {d: (out1_full[:, :d], out2_full[:, :d]) for d in dims}
