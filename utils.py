"""
Utility functions for PL_UCCA experiments.
"""

import os
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def get_total_correlation(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """
    Compute the total correlation (sum of Pearson correlations) between
    two sets of outputs after QR-based orthonormalization.

    Args:
        x1: Tensor of shape (n_samples, dim).
        x2: Tensor of shape (n_samples, dim).

    Returns:
        Total correlation value (float).
    """
    # Center the data #TODO: uncomment this
    # x1 = x1 - x1.mean(dim=0)
    # x2 = x2 - x2.mean(dim=0)

    q_x1, r_x1 = torch.linalg.qr(x1)
    q_x2, r_x2 = torch.linalg.qr(x2)
    x1 = q_x1 @ torch.diag(torch.sign(torch.diag(r_x1)))
    x2 = q_x2 @ torch.diag(torch.sign(torch.diag(r_x2)))

    corr_matrix = x1.T @ x2
    print(f'Correlation by dimension: {torch.diag(corr_matrix)}')
    return torch.diag(corr_matrix).sum().item()


def save_results(name: str, results_dict: dict, results_dir: str = "results") -> str:
    """Save experiment results to a JSON file.

    Args:
        name: Experiment name (used as filename stem).
        results_dict: Dictionary of results to save.
        results_dir: Directory to save results in.

    Returns:
        Path to the saved file.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[results] Saved to {path}")
    return path


def load_results(name: str, results_dir: str = "results") -> dict:
    """Load experiment results from a JSON file."""
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "r") as f:
        return json.load(f)


def get_cross_view_accuracy(train_proj: torch.Tensor, train_labels: torch.Tensor,
                            test_proj: torch.Tensor, test_labels: torch.Tensor,
                            classifier_kind: str = 'linear') -> float:
    """
    Compute cross-view classification accuracy.
    Fits a classifier on one view's projections and tests on the other.

    Args:
        train_proj: Training projections (n_train, dim).
        train_labels: Training labels (n_train).
        test_proj: Testing projections (n_test, dim).
        test_labels: Testing labels (n_test).
        classifier_kind: 'linear' or 'knn'.

    Returns:
        Accuracy (float).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    X_train = train_proj.detach().cpu().numpy()
    y_train = train_labels.detach().cpu().numpy()
    X_test = test_proj.detach().cpu().numpy()
    y_test = test_labels.detach().cpu().numpy()

    if classifier_kind == 'linear':
        clf = LogisticRegression(max_iter=1000)
    elif classifier_kind == 'knn':
        clf = KNeighborsClassifier(n_neighbors=min(5, len(y_train)))
    else:
        raise ValueError(f"Unknown classifier_kind: {classifier_kind}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf.fit(X_train, y_train)
        
    return float(clf.score(X_test, y_test))
