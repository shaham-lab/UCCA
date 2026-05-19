"""
Mini-Vec2Vec runner — same interface as baselines/{sca,cca}/run.py.
"""

import torch
import numpy as np
from sklearn.decomposition import PCA

from ucca.model import qap_cca

def run_qap_cca(X_train, Y_train, output_size, X_test=None, Y_test=None, **kwargs):
    """
    Run QAP-CCA training + transform with PCA pre/post-processing.

    Pre-processing:  If the two views have different dimensionalities, the
                     higher-dimensional view is reduced via PCA to match the
                     lower-dimensional one.
    Post-processing: Both output views are jointly PCA-projected to each
                     requested output dimension.

    Args:
        X_train, Y_train: Training tensors.
        output_size (int or list[int]): Final output dimension(s) via post-PCA.
            If a list, the model is run once and post-PCA is repeated per dim.
        X_test, Y_test: Test tensors (default to train if None).
        **kwargs: Forwarded to qap_cca (n_test, etc.)

    Returns:
        (output1, output2) if output_size is int, else dict[int -> (out1, out2)].
    """
    if X_test is None or Y_test is None:
        X_test = X_train
        Y_test = Y_train

    # ------------------------------------------------------------------
    # Run QAP-CCA (which internally handles post-PCA if output_size is set)
    # ------------------------------------------------------------------
    results = qap_cca(X_train, Y_train, X_test, Y_test, output_size=output_size, **kwargs)

    if isinstance(output_size, list):
        # results is (eval_results, train_results)
        eval_results, train_results = results
        return eval_results
    else:
        # results is (X_output, Y_output, X_train_output, Y_train_output)
        return results[0], results[1]

