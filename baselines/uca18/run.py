import torch
import numpy as np
from baselines.uca18.model import UCA18

def run_uca18(X_train, Y_train, output_size, X_test=None, Y_test=None, **kwargs):
    """
    Run the UCA18 (Hoshen & Wolf 2018) baseline.
    """
    if X_test is None or Y_test is None:
        X_test = X_train
        Y_test = Y_train

    n_subsample = 5000
    if X_train.shape[0] > n_subsample:
        idx = np.random.choice(X_train.shape[0], n_subsample, replace=False)
        X_train = X_train[idx]
        Y_train = Y_train[idx]

    # Parameters
    lr = kwargs.get('lr', 1e-4)
    n_epochs = kwargs.get('n_epochs', 100)
    batch_size = kwargs.get('batch_size', 64)
    lambda_rec = kwargs.get('lambda_rec', 10.0)
    lambda_cyc = kwargs.get('lambda_cyc', 10.0)
    lambda_ortho = kwargs.get('lambda_ortho', 1.0)
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    verbose = kwargs.get('verbose', True)

    input_dim1 = X_train.shape[1]
    input_dim2 = Y_train.shape[1]

    dims = output_size if isinstance(output_size, list) else [output_size]
    results = {}

    # Convert training data to numpy if they are tensors
    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
    Y_train_np = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
    X_test_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
    Y_test_np = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_test

    # For each dimension, we train k independent models and pick the best via consensus
    k = kwargs.get('k_consensus', 5)
    
    for d in dims:
        if verbose:
            print(f"[uca18] Consensus search for latent_dim={d} (k={k})...")
        
        models = []
        scores_list = []
        
        for j in range(k):
            if verbose:
                print(f"  Run {j+1}/{k}...")
            
            model = UCA18(input_dim1, input_dim2, latent_dim=d, 
                          lr=lr, lambda_rec=lambda_rec, lambda_cyc=lambda_cyc, 
                          lambda_ortho=lambda_ortho, batch_size=batch_size, device=device)
            
            model.fit(X_train_np, Y_train_np, n_epochs=n_epochs, verbose=False)
            models.append(model)
            
            # Step 6-11: Compute M[j]
            # M[j][i] = (E1(x_i))^T (E2(tilde_y_i)) where tilde_y_i = D2(E1(x_i))
            model.E1.eval()
            model.E2.eval()
            model.D2.eval()
            
            with torch.no_grad():
                xt = torch.tensor(X_train_np, dtype=torch.float32).to(device)
                z1 = model.E1(xt)
                y_tilde = model.D2(z1)
                z2_tilde = model.E2(y_tilde)
                
                # Dot product per sample
                scores = torch.sum(z1 * z2_tilde, dim=1).cpu().numpy()
                scores_list.append(scores)
        
        # Step 12: Consensus via SVD
        M = np.stack(scores_list, axis=1) # (n_samples, k)
        M_centered = M - M.mean(axis=0)
        
        u, s, vh = np.linalg.svd(M_centered, full_matrices=False)
        P = vh[0] # First right singular vector (coefficients for the runs)
        
        # Step 13: Best run
        j_star = np.argmax(P)
        
        if verbose:
            print(f"  Consensus selected run {j_star+1} with weight {P[j_star]:.4f}")
            
        best_model = models[j_star]
        out1, out2 = best_model.transform(X_test_np, Y_test_np)
        results[d] = (out1, out2)

    if isinstance(output_size, list):
        return results
    else:
        return results[output_size]
