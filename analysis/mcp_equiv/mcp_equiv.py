import argparse
import yaml
import torch
import numpy as np
import os
import json
import sys

# Add the project root to sys.path to allow imports from root modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data import get_data, prepare_data
from ucca.model import qap_cca
from ucca.qap_cca_utils import cos_sim_matrix
from analysis.mcp_equiv.distances import get_all_distances
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

def generate_random_distributions(n_clusters_list, n_repeats):
    os.makedirs('results/mcp_equiv', exist_ok=True)
    for K in n_clusters_list:
        print(f"Generating random distribution for K={K} ({n_repeats} repeats)...")
        results = []
        for _ in tqdm(range(n_repeats), desc=f'K={K}'):
            p = np.random.permutation(K)
            dist = get_all_distances(p)
            dist['tc'] = 0.0
            results.append(dist)
        
        output_path = f'results/mcp_equiv/random_distances_{K}.json'
        # For backward compatibility, if K=20, also save as random_distances.json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        if K == 20:
            with open('results/mcp_equiv/random_distances.json', 'w') as f:
                json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=False, help='Data kind (e.g., handwritten13)')
    parser.add_argument('--n_repeats', type=int, default=100, help='Number of repeats')
    parser.add_argument('--random', action='store_true', help='Generate random distributions')
    parser.add_argument('--n_clusters', type=int, nargs='+', default=[20, 12], help='Number of clusters for random distributions')
    parser.add_argument('--n_repeats_random', type=int, default=1000, help='Number of repeats for random distributions')
    args = parser.parse_args()

    if args.random:
        generate_random_distributions(args.n_clusters, args.n_repeats_random)
        if not args.data:
            return

    if not args.data:
        print("Error: --data is required unless --random is specified.")
        sys.exit(1)

    config_path = f'configs/default/ucca_{args.data}.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_kind = config['data']
    view1_idx = config.get('view1_idx')
    view2_idx = config.get('view2_idx')
    n_test = config.get('n_test', 256)
    
    # Load data
    A, B, _ = get_data(data_kind, view1_idx=view1_idx, view2_idx=view2_idx)
    
    results = []
    
    print(f"Running MCP Equivalence Analysis for {data_kind}...")
    for i in tqdm(range(args.n_repeats), desc='Repeats'):
        # Prepare data (shuffles and splits)
        X_train, X_test, Y_train, Y_test, _, _, _ = prepare_data(A, B, n_test=n_test)
        
        # Run qap_cca with anchors_n_runs=1 and return_clusters=True
        # We use the parameters from the config but override anchors_n_runs
        X_out, Y_out, X_tr_out, Y_tr_out, c1, c2 = qap_cca(
            X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy(),
            qap_solver=config.get('qap_solver', '2opt'),
            qap_n_runs=config.get('qap_n_runs', 30),
            n_anchors=config.get('n_anchors', 20),
            qap_subsample=config.get('qap_subsample', 10000),
            anchors_n_runs=1,
            pca_n_components=config.get('pca_n_components'),
            projection_type=config.get('projection_type', 'procrustes'),
            return_clusters=True
        )
        
        # TC validation
        # Compute cosine similarity between projected test sets
        # Convert to torch if needed
        X_out_t = torch.as_tensor(X_out).float()
        Y_out_t = torch.as_tensor(Y_out).float()
        tc = torch.cosine_similarity(X_out_t, Y_out_t, dim=-1).mean().item()
        print(f"Repeat {i}: Total Correlation = {tc:.4f}")
        
        # c1 and c2 are matched clusters (K x d)
        # mcp_sf is the identity (0, 1, ..., K-1)
        # Find mcp_od using Procrustes and Correlation
        c1_np = c1.cpu().numpy()
        c2_np = c2.cpu().numpy()
        
        # Q such that c1 @ Q approx c2
        Q, _ = orthogonal_procrustes(c1_np, c2_np)
        Q_t = torch.tensor(Q).float().to(c1.device)
        
        # Cost matrix = - cosine similarity matrix
        cost_matrix = -cos_sim_matrix(c1 @ Q_t, c2).cpu().numpy()
        # Find permutation mcp_od that minimizes cost (maximizes correlation)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Compute distances between identity and col_ind
        dist = get_all_distances(col_ind)
        dist['tc'] = tc
        results.append(dist)

    # Save results
    os.makedirs('results/mcp_equiv', exist_ok=True)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    output_path = f'results/mcp_equiv/{config_name}_distances.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()

