import torch
import numpy as np
import os
import sys
import argparse

# Add the project root to the python path to import get_data
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from data import get_data

def apply_random_swaps(perm, n_shuffles):
    """Apply n_shuffles random swaps to the permutation array."""
    n = len(perm)
    if n_shuffles == 0:
        return perm
    
    # Generate random pairs of indices to swap
    # Using numpy for random choice to avoid potential torch seed issues in loops if not careful
    idx_pairs = np.random.randint(0, n, size=(n_shuffles, 2))
    
    for i in range(n_shuffles):
        i1, i2 = idx_pairs[i]
        perm[i1], perm[i2] = perm[i2], perm[i1]
        
    return perm

def whiten(X, tol=1e-9):
    """Robust whitening using double precision and rank detection."""
    # Convert to float64 for numerical stability with large datasets (e.g. QM9)
    X = X.to(torch.float64)
    # Explicit centering
    X = X - X.mean(dim=0)
    n = X.shape[0]
    
    # Compute covariance matrix in double precision
    sigma = (X.T @ X) / n
    # Eigendecomposition (sigma is symmetric)
    vals, vecs = torch.linalg.eigh(sigma)
    
    # Detect and handle rank deficiency
    max_val = vals.max()
    mask = vals > (max_val * tol)
    
    vals_filt = vals[mask]
    vecs_filt = vecs[:, mask]
    
    if len(vals_filt) == 0:
        return torch.zeros_like(X)
    
    # Linear whitening transformation
    # W = E * D^{-1/2} * E^T
    inv_sqrt = vecs_filt @ torch.diag(1.0 / torch.sqrt(vals_filt)) @ vecs_filt.T
    
    return X @ inv_sqrt

def run_analysis(args):
    dataset_name = args.dataset
    view1_idx = args.view1_idx
    view2_idx = args.view2_idx
    
    save_name = dataset_name
    if view1_idx is not None and view2_idx is not None:
        save_name = f"{dataset_name}_{view1_idx}_{view2_idx}"
        
    print(f"Loading {dataset_name} dataset...")
    if view1_idx is not None and view2_idx is not None:
        X, Y, _ = get_data(dataset_name, view1_idx=view1_idx, view2_idx=view2_idx)
    else:
        X, Y, _ = get_data(dataset_name)
    
    n_samples = X.shape[0]
    print(f"Dataset size: {n_samples} samples")
    
    # Pre-process: Robust Whitening
    print("Whitening views (Double Precision)...")
    # Whitened versions for correlation
    X_white = whiten(X)
    Y_white = whiten(Y)
    
    n_shuffles_list = [0, 10, 100, 500, 1000, 2000, 5000]
    # n_shuffles_list = [0, 10, 100, 1000, 5000, 10000, 20000, 50000, 100000] # for QM9
    n_repeats = args.n_repeats
    
    results = {
        'n_shuffles_list': n_shuffles_list,
        'n_repeats': n_repeats,
        'dataset': dataset_name,
        'save_name': save_name,
        'view1_idx': view1_idx,
        'view2_idx': view2_idx,
        'svs_corr': {}
    }
    
    for n_shuffles in n_shuffles_list:
        print(f"Running n_shuffles = {n_shuffles}...")
        batch_corr_svs = []
        
        for r in range(n_repeats):
            perm = np.arange(n_samples)
            perm = apply_random_swaps(perm, n_shuffles)
            
            # Correlation / Whitened Covariance (Double Precision)
            Yw_shuff = Y_white[perm]
            corr = (X_white.T @ Yw_shuff) / n_samples
            
            # Singular values should be <= 1 theoretically.
            # Double precision avoids numerical overshoot.
            svs_corr = torch.linalg.svdvals(corr)
            batch_corr_svs.append(svs_corr.cpu().numpy().astype(np.float32))
            
        results['svs_corr'][n_shuffles] = np.array(batch_corr_svs)
        
    # Save results to analysis/cov_svd/results.pt
    output_dir = os.path.join(project_root, 'results/cov_svd', save_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'results.pt')
    
    torch.save(results, output_path)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SVD of covariance and correlation under shuffling.")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., qm9, handwritten)")
    parser.add_argument("view1_idx", type=int, nargs='?', default=None, help="Index of view 1")
    parser.add_argument("view2_idx", type=int, nargs='?', default=None, help="Index of view 2")
    parser.add_argument("--n_repeats", type=int, default=10, help="Number of repetitions per shuffle level")
    parser.add_argument("--corr", action="store_true", help="Ignored (for compatibility with plot_cov_svd.py args)")
    
    args = parser.parse_args()
    
    # Ensure reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    run_analysis(args)
