import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA


def run_pca_cca(X_train, Y_train, output_size, X_test=None, Y_test=None, pca_dim=None, **kwargs):
    """
    Run PCA + CCA.

    Args:
        X_train, Y_train: Training tensors.
        output_size (int or list[int]): Number of CCA components. If a list,
            CCA is fit once with max(output_size) and results are returned
            for each dim as a dict {dim: (out1, out2)}.
        X_test, Y_test: Test tensors (default to train if None).
        pca_dim (int, optional): Explicit number of PCA dimensions to reduce both views to before CCA.

    Returns:
        (output1, output2) if output_size is int, else dict[int -> (out1, out2)].
    """
    if X_test is None or Y_test is None:
        X_test = X_train
        Y_test = Y_train

    dims = output_size if isinstance(output_size, list) else None
    fit_size = max(dims) if dims is not None else output_size

    # ------------------------------------------------------------------
    # Pre-PCA: reduce the higher-dim view to match the lower-dim view
    # ------------------------------------------------------------------
    dim_x, dim_y = X_train.shape[1], Y_train.shape[1]

    if pca_dim is not None:
        pca_pre_x = PCA(n_components=pca_dim)
        X_train = pca_pre_x.fit_transform(X_train.cpu().numpy() if hasattr(X_train, 'cpu') else X_train)
        X_test  = pca_pre_x.transform(X_test.cpu().numpy() if hasattr(X_test, 'cpu') else X_test)
        
        pca_pre_y = PCA(n_components=pca_dim)
        Y_train = pca_pre_y.fit_transform(Y_train.cpu().numpy() if hasattr(Y_train, 'cpu') else Y_train)
        Y_test  = pca_pre_y.transform(Y_test.cpu().numpy() if hasattr(Y_test, 'cpu') else Y_test)
    elif dim_x != dim_y:
        target_dim = min(dim_x, dim_y)

        if dim_x > dim_y:
            pca_pre = PCA(n_components=target_dim)
            X_train = pca_pre.fit_transform(X_train.cpu().numpy() if hasattr(X_train, 'cpu') else X_train)
            X_test  = pca_pre.transform(X_test.cpu().numpy() if hasattr(X_test, 'cpu') else X_test)
            
            # ensure Y is numpy for consistency with CCA
            Y_train = Y_train.cpu().numpy() if hasattr(Y_train, 'cpu') else Y_train
            Y_test = Y_test.cpu().numpy() if hasattr(Y_test, 'cpu') else Y_test
        else:
            pca_pre = PCA(n_components=target_dim)
            Y_train = pca_pre.fit_transform(Y_train.cpu().numpy() if hasattr(Y_train, 'cpu') else Y_train)
            Y_test  = pca_pre.transform(Y_test.cpu().numpy() if hasattr(Y_test, 'cpu') else Y_test)
            
            # ensure X is numpy for consistency with CCA
            X_train = X_train.cpu().numpy() if hasattr(X_train, 'cpu') else X_train
            X_test = X_test.cpu().numpy() if hasattr(X_test, 'cpu') else X_test
    else:
        # ensure numpy
        X_train = X_train.cpu().numpy() if hasattr(X_train, 'cpu') else X_train
        X_test = X_test.cpu().numpy() if hasattr(X_test, 'cpu') else X_test
        Y_train = Y_train.cpu().numpy() if hasattr(Y_train, 'cpu') else Y_train
        Y_test = Y_test.cpu().numpy() if hasattr(Y_test, 'cpu') else Y_test

    # ------------------------------------------------------------------
    # Run CCA (fit once with max components)
    # ------------------------------------------------------------------
    cca = CCA(n_components=fit_size)
    cca.fit(X_train, Y_train)

    out1_full, out2_full = cca.transform(X_test, Y_test)

    if dims is None:
        return out1_full, out2_full

    return {d: (out1_full[:, :d], out2_full[:, :d]) for d in dims}
