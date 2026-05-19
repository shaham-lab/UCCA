from tqdm.auto import trange, tqdm
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans
import torch
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

from ucca.qap_cca_utils import N, sim, tensor, cos_sim_matrix, eval_score
from ucca.qap_cca_utils import train_orthogonal_linear_procrustes, train_orthogonal_linear_cca


def aligned_centroids(X_train, Y_train, qap_n_runs=300, n_anchors=50, qap_anchor_kind='kmeans', qap_solver='2opt', subsample=None, n_jobs=-1):
    options = {'P0': 'randomized', 'maximize': True}
    if subsample is not None:
        if qap_anchor_kind.startswith('paired_'):
            perm = torch.randperm(len(X_train))[:subsample]
            X_train, Y_train = X_train[perm], Y_train[perm]
        else:
            perm_x = torch.randperm(len(X_train))[:subsample]
            perm_y = torch.randperm(len(Y_train))[:subsample]
            X_train, Y_train = X_train[perm_x], Y_train[perm_y]

    if qap_anchor_kind == 'kmeans':
        clusterer1 = KMeans(n_clusters=n_anchors, n_init='auto')
        clusterer1.fit(X_train)
        centers1 = clusterer1.cluster_centers_
        clusterer2 = KMeans(n_clusters=n_anchors, n_init='auto')
        clusterer2.fit(Y_train)
        centers2 = clusterer2.cluster_centers_
    elif qap_anchor_kind == 'random':
        indices1 = np.random.choice(len(X_train), n_anchors, replace=False)
        centers1 = X_train[indices1].cpu().numpy()
        indices2 = np.random.choice(len(Y_train), n_anchors, replace=False)
        centers2 = Y_train[indices2].cpu().numpy()
    elif qap_anchor_kind == 'paired_kmeans':
        clusterer1 = KMeans(n_clusters=n_anchors, n_init='auto')
        clusterer1.fit(X_train)
        centroids = torch.tensor(clusterer1.cluster_centers_, device=X_train.device, dtype=X_train.dtype)
        dists = torch.cdist(centroids, X_train)
        indices = torch.argmin(dists, dim=1)
        centers1 = X_train[indices].cpu().numpy()
        centers2 = Y_train[indices].cpu().numpy()
    elif qap_anchor_kind == 'paired_kmeans_reversed':
        clusterer2 = KMeans(n_clusters=n_anchors, n_init='auto')
        clusterer2.fit(Y_train)
        centroids = torch.tensor(clusterer2.cluster_centers_, device=Y_train.device, dtype=Y_train.dtype)
        dists = torch.cdist(centroids, Y_train)
        indices = torch.argmin(dists, dim=1)
        centers1 = X_train[indices].cpu().numpy()
        centers2 = Y_train[indices].cpu().numpy()
    else:
        raise ValueError(f"Unknown anchor kind: {qap_anchor_kind}")

    kernel1 =  sim(centers1, centers1).float()
    kernel2 = sim(centers2, centers2).float()

    # need to re-run the QAP a few times because it's not very good at finding the global optimum (even 2opt)
    results = Parallel(n_jobs=n_jobs)(
        delayed(quadratic_assignment)(kernel1, kernel2, method=qap_solver, options=options)
        for _ in trange(qap_n_runs, desc='QAP runs', leave=False)
    )
    quad = max(results, key=lambda x: x.fun)
    centers2 = centers2[quad.col_ind]
    return tensor(centers1), tensor(centers2)


class UCCA:
    def __init__(self, output_size=None, qap_solver='2opt', qap_n_runs=30, n_anchors=20, qap_anchor_kind='kmeans', qap_subsample=10_000, anchors_n_runs=500, pca_n_components=None, n_jobs=-1, projection_type='cca'):
        self.output_size = output_size
        self.qap_solver = qap_solver
        self.qap_n_runs = qap_n_runs
        self.n_anchors = n_anchors
        self.qap_anchor_kind = qap_anchor_kind
        self.qap_subsample = qap_subsample
        self.anchors_n_runs = anchors_n_runs
        self.pca_n_components = pca_n_components
        self.n_jobs = n_jobs
        self.projection_type = projection_type
        
        # State variables
        self.pca_X = None
        self.pca_Y = None
        self.W_X = None
        self.W_Y = None
        self.all_centers1 = None
        self.all_centers2 = None
        self.pcas_post = None
        self.X_mean = 0
        self.Y_mean = 0
        self.X_std = 1
        self.Y_std = 1

    def fit(self, X_train, Y_train):
        # PCA-whitening
        if self.pca_n_components is not None:
            self.pca_X = PCA(n_components=self.pca_n_components)
            X_train = self.pca_X.fit_transform(X_train)
            self.pca_Y = PCA(n_components=self.pca_n_components)
            Y_train = self.pca_Y.fit_transform(Y_train)

        X_train = tensor(X_train)
        Y_train = tensor(Y_train)

        # Step 1: Match Anchors
        print('Matching anchors...')
        all_centers1, all_centers2 = [], []
        for i in trange(self.anchors_n_runs):
            centers1, centers2 = aligned_centroids(X_train, Y_train, subsample=self.qap_subsample,
                                                n_anchors=self.n_anchors, qap_anchor_kind=self.qap_anchor_kind, qap_n_runs=self.qap_n_runs, qap_solver=self.qap_solver, n_jobs=self.n_jobs)
            all_centers1.append(centers1)
            all_centers2.append(centers2)

        self.all_centers1 = torch.cat(all_centers1, dim=0)
        self.all_centers2 = torch.cat(all_centers2, dim=0)

        # Step 2: Train Mapping
        print('Training mapping...')
        if self.projection_type == 'procrustes':
            self.W_X = train_orthogonal_linear_procrustes(self.all_centers1, self.all_centers2)
            self.W_Y = torch.eye(self.W_X.shape[1], device=self.W_X.device)
        elif self.projection_type == 'cca':
            self.W_X, self.W_Y = train_orthogonal_linear_cca(self.all_centers1, self.all_centers2)
        else:
            raise ValueError(f'Unknown projection type: {self.projection_type}')

        X_train_out = X_train @ self.W_X
        Y_train_out = Y_train @ self.W_Y

        X_train_out = X_train_out.cpu().numpy()
        Y_train_out = Y_train_out.cpu().numpy()

        stacked = np.concatenate([X_train_out, Y_train_out], axis=0)

        self.pcas_post = {}
        if self.output_size is not None:
            dims = self.output_size if isinstance(self.output_size, list) else [self.output_size]
            for d in dims:
                pca_post = PCA(n_components=d)
                pca_post.fit(stacked)
                self.pcas_post[d] = pca_post

        self.X_mean = X_train_out.mean(axis=0)
        self.Y_mean = Y_train_out.mean(axis=0)
        self.X_std = X_train_out.std(axis=0)
        self.Y_std = Y_train_out.std(axis=0)

        return self

    def transform(self, X, Y):
        if self.pca_X is not None:
            X = self.pca_X.transform(X)
        if self.pca_Y is not None:
            Y = self.pca_Y.transform(Y)
        
        X = tensor(X)
        Y = tensor(Y)
        
        X_output = X @ self.W_X
        Y_output = Y @ self.W_Y

        X_output = X_output.cpu().numpy()
        Y_output = Y_output.cpu().numpy()

        X_output = (X_output - self.X_mean) / (self.X_std + 1e-8)
        Y_output = (Y_output - self.Y_mean) / (self.Y_std + 1e-8)

        if self.output_size is not None and self.pcas_post is not None:
            if isinstance(self.output_size, list):
                results = {}
                for d in self.output_size:
                    results[d] = (tensor(self.pcas_post[d].transform(X_output)), tensor(self.pcas_post[d].transform(Y_output)))
                return results
            else:
                return tensor(self.pcas_post[self.output_size].transform(X_output)), tensor(self.pcas_post[self.output_size].transform(Y_output))

        return tensor(X_output), tensor(Y_output)

    def fit_transform(self, X_train, Y_train):
        self.fit(X_train, Y_train)
        return self.transform(X_train, Y_train)


def qap_cca(X_train, Y_train, X_eval, Y_eval, output_size=None, qap_solver='2opt', qap_n_runs=30, n_anchors=20, qap_anchor_kind='kmeans', qap_subsample=10_000, anchors_n_runs=500, pca_n_components=None, n_jobs=-1, projection_type='cca', return_clusters=False):
    model = UCCA(
        output_size=output_size, qap_solver=qap_solver, qap_n_runs=qap_n_runs, n_anchors=n_anchors, 
        qap_anchor_kind=qap_anchor_kind, qap_subsample=qap_subsample, 
        anchors_n_runs=anchors_n_runs, pca_n_components=pca_n_components, 
        n_jobs=n_jobs, projection_type=projection_type
    )
    
    train_results = model.fit_transform(X_train, Y_train)
    eval_results = model.transform(X_eval, Y_eval)
    
    if isinstance(output_size, list):
        if return_clusters:
            return eval_results, train_results, model.all_centers1, model.all_centers2
        return eval_results, train_results
    else:
        X_train_output, Y_train_output = train_results
        X_output, Y_output = eval_results
        
        if return_clusters:
            return X_output, Y_output, X_train_output, Y_train_output, model.all_centers1, model.all_centers2
        return X_output, Y_output, X_train_output, Y_train_output