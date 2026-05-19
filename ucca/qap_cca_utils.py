import numpy as np
from tqdm.auto import trange, tqdm
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
import os

def cos_sim_matrix(X, Y):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)
    X_norm = X / X.norm(dim=-1, keepdim=True)
    Y_norm = Y / Y.norm(dim=-1, keepdim=True)
    return X_norm @ Y_norm.T

def tensor(x):
    return torch.tensor(x).float()

def N(X, dim=-1, **kwargs):
    return F.normalize(X, dim=dim, **kwargs)

def sim(X, Y):
    X, Y = tensor(X), tensor(Y)
    # center the tensors
    # TODO: centering probably not necessary, and perhaps not implemented correctly, might need transpose somewhere and stuff
    H = torch.eye(len(X), device=X.device) - (1/len(X)) * torch.ones((len(X), len(X)), device=X.device)
    return H @ X @ Y.T @ H

def train_orthogonal_linear_procrustes(X, Y):
    solution, _ = orthogonal_procrustes(X, Y)
    return tensor(solution)

def train_orthogonal_linear_cca(X, Y):
    cca = CCA(n_components=X.shape[1])
    cca.fit(X, Y)
    return tensor(cca.x_weights_), tensor(cca.y_weights_)

def eval_score(X_eval, Y_eval, W, backward=False):
    if backward:
        return torch.round(torch.cosine_similarity(X_eval, Y_eval @ W.T, dim=-1).mean(), decimals=2)
    else:
        return torch.round(torch.cosine_similarity(X_eval @ W, Y_eval, dim=-1).mean(), decimals=2)

def rank(X):
    return torch.argsort(torch.argsort(X, dim=-1), dim=-1)

def get_metrics(X, Y):
    ranks = rank(cos_sim_matrix(X, Y)).diagonal()
    top1acc = (len(X) - 1 == ranks).float().mean().item()
    avg_rank = len(X) - ranks.float().mean().item()
    return {"top1acc": top1acc, "avg_rank": avg_rank}
