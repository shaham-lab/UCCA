"""
Self-contained data loading module for PL_UCCA experiments.

Supports: handwritten, snare, coco, flickr8k.
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Individual dataset loaders
# ---------------------------------------------------------------------------

def get_handwritten(view1_idx: int = 1, view2_idx: int = 3,
                    shuffle: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the UCI Multi-feature handwritten digits dataset."""
    from mvlearn.datasets import load_UCImultifeature
    mv_data, labels = load_UCImultifeature()
    A = torch.tensor(mv_data[view1_idx], dtype=torch.float32)
    B = torch.tensor(mv_data[view2_idx], dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    if shuffle:
        B = B[torch.randperm(B.shape[0])]
    return A, B, labels

def get_snare() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load snare embeddings."""
    from sklearn.preprocessing import normalize
    X1 = np.load("datasets/snare/SNAREseq_atac_feat.npy")
    X2 = np.load("datasets/snare/SNAREseq_rna_feat.npy")
    labels = np.loadtxt("datasets/snare/SNAREseq_atac_types.txt")
    X1 = normalize(X1, axis=1)
    X2 = normalize(X2, axis=1)
    return (torch.tensor(X1).type(torch.FloatTensor), 
            torch.tensor(X2).type(torch.FloatTensor),
            torch.tensor(labels).type(torch.LongTensor))

def get_coco() -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load coco embeddings and labels."""
    d = 'datasets/coco/'
    m1, m2 = 'encoded1', 'encoded2'
    labels_path = f'{d}labels.pt'
    labels = torch.load(labels_path, map_location='cpu') if os.path.exists(labels_path) else None
    return (torch.load(f'{d}{m1}.pt', map_location='cpu'),
            torch.load(f'{d}{m2}.pt', map_location='cpu'),
            labels)

def get_flickr8k() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load flickr8k embeddings."""
    d = 'datasets/flickr8/'
    m1, m2 = 'encoded1', 'encoded2'
    return torch.load(f'{d}{m1}.pt', map_location='cpu'), torch.load(f'{d}{m2}.pt', map_location='cpu')


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def get_data(data_kind: str, n_points: Optional[int] = None,
             view1_idx=None, view2_idx=None,
             shuffle: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Load a dataset by name.

    Args:
        data_kind: One of 'handwritten', 'snare', 'coco', 'flickr8k'.
        n_points:  Number of samples (unused, for compat).
        view1_idx: Index/key for the first view.
        view2_idx: Index/key for the second view.
        shuffle:   Whether to shuffle the second view.

    Returns:
        (embed_A, embed_B, labels): Two tensors of shape (n_samples, dim) and optional labels.
    """
    labels = None
    if data_kind == 'handwritten':
        v1 = int(view1_idx) if view1_idx is not None else 1
        v2 = int(view2_idx) if view2_idx is not None else 3
        A, B, labels = get_handwritten(v1, v2, shuffle=shuffle)
    elif data_kind == 'snare':
        A, B, labels = get_snare()
    elif data_kind == 'coco':
        A, B, labels = get_coco()
    elif data_kind == 'flickr8k':
        A, B = get_flickr8k()
    else:
        raise ValueError(f"Unknown data_kind: {data_kind}")

    print(f"[data] {data_kind}: view1 {A.shape}, view2 {B.shape}")
    return A, B, labels


def prepare_data(embed_A: torch.Tensor, embed_B: torch.Tensor,
                 n_test: int, split: str = "unpaired", 
                 labels: Optional[torch.Tensor] = None):
    """
    Shuffle and split into train / test, center by training mean.

    Args:
        split: 'paired' — X_train and Y_train share the same indices.
               'unpaired' (default) — Y_train uses different indices than X_train.
        labels: Optional labels for the data.

    Returns:
        (X_train, X_test, Y_train, Y_test, L_X_train, L_Y_train, L_test)
    """
    perm = torch.randperm(len(embed_A))
    embed_A = embed_A[perm]
    embed_B = embed_B[perm]
    if labels is not None:
        labels = labels[perm]

    n_train = (len(embed_A) - n_test) // 2

    L_X_train, L_Y_train, L_test = None, None, None

    if split == "paired":
        X_train = embed_A[:n_train]
        X_test = embed_A[n_train*2:]
        Y_train = embed_B[:n_train]
        Y_test = embed_B[n_train*2:]
        if labels is not None:
            L_X_train = labels[:n_train]
            L_Y_train = labels[:n_train]
            L_test = labels[n_train*2:]
    else:
        X_train = embed_A[:n_train]
        X_test = embed_A[n_train*2:]
        Y_train = embed_B[n_train:n_train*2]
        Y_test = embed_B[n_train*2:]
        if labels is not None:
            L_X_train = labels[:n_train]
            L_Y_train = labels[n_train:n_train*2]
            L_test = labels[n_train*2:]

    # Center by training mean
    mean_A = X_train.mean(dim=0)
    mean_B = Y_train.mean(dim=0)
    X_train = X_train - mean_A
    X_test = X_test - mean_A
    Y_train = Y_train - mean_B
    Y_test = Y_test - mean_B

    return X_train, X_test, Y_train, Y_test, L_X_train, L_Y_train, L_test
