"""
Unsupervised Shared Component Analysis (USCA) Model

A standalone implementation of the USCA model for cross-domain representation learning.
The model learns shared representations between two domains using adversarial training
with orthogonality constraints and optional supervision.

Usage:
    model = USCA(D=256, orthogonal_w=1.0, supervised_w=10.0)
    model.fit(X_source, X_target, y_source=None, y_target=None, 
              num_anchors=256, n_epochs=76)
    c1, c2 = model.transform(X_source, X_target)
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import NearestNeighbors
import warnings
from tqdm import trange



class Discriminator(nn.Module):
    """Discriminator network for adversarial training."""
    
    def __init__(self, input_size: int, output_size: int = 1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class ForeverDataIterator:
    """A data iterator that will never stop producing data."""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


class USCA:
    """
    Unsupervised Shared Component Analysis model.
    
    This model learns shared representations between two domains using adversarial training
    with orthogonality constraints and optional supervision via anchor points.
    
    Parameters:
    -----------
    D : int, default=256
        Number of shared components (latent dimension)
    orthogonal_w : float, default=1.0
        Weight for orthogonality constraint
    supervised_w : float, default=10.0
        Weight for supervised loss (anchor points)
    encoder_lr : float, default=1e-3
        Learning rate for encoder matrices Z1, Z2
    discr_lr : float, default=1e-4
        Learning rate for discriminator
    batch_size : int, default=32
        Batch size for training
    n_z : int, default=1
        Number of encoder updates per discriminator update
    lsmooth : float, default=1.0
        Label smoothing factor
    device : str, default='auto'
        Device to use ('cuda', 'cpu', or 'auto')
    verbose : bool, default=True
        Whether to print training progress
    """
    
    def __init__(self, 
                 D: int = 256,
                 orthogonal_w: float = 1.0,
                 supervised_w: float = 10.0,
                 encoder_lr: float = 1e-3,
                 discr_lr: float = 1e-4,
                 batch_size: int = 32,
                 n_z: int = 1,
                 lsmooth: float = 1.0,
                 device: str = 'auto',
                 verbose: bool = True,
                 loss_func: str = 'CE'):
        
        self.D = D
        self.orthogonal_w = orthogonal_w
        self.supervised_w = supervised_w
        self.encoder_lr = encoder_lr
        self.discr_lr = discr_lr
        self.batch_size = batch_size
        self.n_z = n_z
        self.lsmooth = lsmooth
        self.verbose = verbose
        
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model components
        self.Z1 = None
        self.Z2 = None
        self.f = None
        self.G1 = None
        self.G2 = None
        self.ID = None
        
        # Training data
        self.v1 = None
        self.v2 = None
        self.anchor1 = None
        self.anchor2 = None
        
        # Optimizers
        self.optimizer_z1 = None
        self.optimizer_z2 = None
        self.optimizer_f = None
        
        # Loss function
        self.loss_func = nn.BCELoss()    

        # Training history
        self.training_history = {
            'cca_losses': [],
            'f_losses': [],
            'z_losses': []
        }
        
        if self.verbose:
            print(f"USCA model initialized with D={D}, device={self.device}")
    
    def _prepare_data(self, X_source: np.ndarray, X_target: np.ndarray,
                     y_source: Optional[np.ndarray] = None,
                     y_target: Optional[np.ndarray] = None) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders from input arrays."""
        
        # Convert to tensors
        X_source = torch.FloatTensor(X_source)
        X_target = torch.FloatTensor(X_target)
        
        # Create dummy labels if not provided
        if y_source is None:
            y_source = torch.zeros(X_source.shape[0], dtype=torch.long)
        else:
            y_source = torch.LongTensor(y_source)
            
        if y_target is None:
            y_target = torch.zeros(X_target.shape[0], dtype=torch.long)
        else:
            y_target = torch.LongTensor(y_target)
        
        # Create datasets and loaders
        source_dataset = TensorDataset(X_source, y_source)
        target_dataset = TensorDataset(X_target, y_target)
        
        source_loader = DataLoader(source_dataset, batch_size=self.batch_size, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=self.batch_size, shuffle=True)
        
        return source_loader, target_loader
    
    def _compute_covariance_matrices(self, source_loader: DataLoader, target_loader: DataLoader):
        """Compute covariance matrices G1 and G2 for orthogonality constraints."""
        
        if self.verbose:
            print("Computing covariance matrices...")
        
        src_size = source_loader.dataset[0][0].shape[-1]
        tgt_size = target_loader.dataset[0][0].shape[-1]
        
        G1 = torch.zeros((src_size, src_size))
        G2 = torch.zeros((tgt_size, tgt_size))
        
        N1 = 0
        N2 = 0
        
        # Compute for source domain
        for batch in source_loader:
            x_s = batch[0]
            G1 += x_s.T @ x_s
            N1 += x_s.shape[0]
        
        # Compute for target domain
        for batch in target_loader:
            x_t = batch[0]
            G2 += x_t.T @ x_t
            N2 += x_t.shape[0]
        
        # Normalize
        G1 = (1/N1) * G1
        G2 = (1/N2) * G2
        
        return G1.to(self.device), G2.to(self.device)
    
    def _prepare_validation_data(self, source_loader: DataLoader, target_loader: DataLoader):
        """Prepare validation data for computing CCA loss."""
        
        v1_list = []
        v2_list = []
        
        # Collect all data
        for batch in source_loader:
            v1_list.append(batch[0])
        
        for batch in target_loader:
            v2_list.append(batch[0])
        
        v1 = torch.cat(v1_list, dim=0).to(self.device)
        v2 = torch.cat(v2_list, dim=0).to(self.device)
        
        # Balance the datasets (take minimum number of samples)
        min_samples = min(v1.shape[0], v2.shape[0])
        v1 = v1[:min_samples]
        v2 = v2[:min_samples]
        
        return v1, v2
    
    def _prepare_anchor_points(self, source_loader: DataLoader, target_loader: DataLoader, 
                              num_anchors: int):
        """Prepare anchor points for supervised learning."""
        
        if num_anchors <= 0:
            return None, None
        
        # Collect all data
        all_source = torch.cat([batch[0] for batch in source_loader], dim=0)
        all_target = torch.cat([batch[0] for batch in target_loader], dim=0)
        
        # Randomly sample anchor points
        min_samples = min(all_source.shape[0], all_target.shape[0])
        num_anchors = min(num_anchors, min_samples)
        
        indices = torch.randperm(min_samples)[:num_anchors]
        
        anchor1 = all_source[indices].to(self.device)
        anchor2 = all_target[indices].to(self.device)
        
        return anchor1, anchor2
    
    def _initialize_model(self, src_size: int, tgt_size: int):
        """Initialize model parameters."""
        
        # Initialize encoder matrices
        self.Z1 = math.sqrt(2/src_size) * torch.randn(self.D, src_size, device=self.device)
        self.Z2 = math.sqrt(2/tgt_size) * torch.randn(self.D, tgt_size, device=self.device)
        self.Z1.requires_grad = True
        self.Z2.requires_grad = True
        
        # Initialize discriminator
        self.f = Discriminator(self.D, 1).to(self.device)
        
        # Identity matrix for orthogonality constraint
        self.ID = torch.eye(self.D, device=self.device)
        
        # Initialize optimizers
        self.optimizer_z1 = torch.optim.Adam([self.Z1], lr=self.encoder_lr)
        self.optimizer_z2 = torch.optim.Adam([self.Z2], lr=self.encoder_lr)
        self.optimizer_f = torch.optim.Adam(self.f.parameters(), lr=self.discr_lr)
        
        if self.verbose:
            print(f"Model initialized: src_size={src_size}, tgt_size={tgt_size}")
    
    def fit(self, X_source: Union[np.ndarray, torch.Tensor], 
            X_target: Union[np.ndarray, torch.Tensor],
            y_source: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_target: Optional[Union[np.ndarray, torch.Tensor]] = None,
            num_anchors: int = 256,
            n_epochs: int = 76) -> 'USCA':
        """
        Fit the USCA model to the data.
        
        Parameters:
        -----------
        X_source : array-like, shape (n_samples, n_features)
            Source domain data
        X_target : array-like, shape (n_samples, n_features)
            Target domain data
        y_source : array-like, shape (n_samples,), optional
            Source domain labels
        y_target : array-like, shape (n_samples,), optional
            Target domain labels
        num_anchors : int, default=256
            Number of anchor points for supervision
        n_epochs : int, default=76
            Number of training epochs
            
        Returns:
        --------
        self : USCA
            Returns self for method chaining
        """
        
        # Convert to numpy arrays if needed
        if isinstance(X_source, torch.Tensor):
            X_source = X_source.numpy()
        if isinstance(X_target, torch.Tensor):
            X_target = X_target.numpy()
        if y_source is not None and isinstance(y_source, torch.Tensor):
            y_source = y_source.numpy()
        if y_target is not None and isinstance(y_target, torch.Tensor):
            y_target = y_target.numpy()
        
        # Prepare data loaders
        source_loader, target_loader = self._prepare_data(X_source, X_target, y_source, y_target)
        
        # Get data dimensions
        src_size = X_source.shape[1]
        tgt_size = X_target.shape[1]
        
        # Initialize model
        self._initialize_model(src_size, tgt_size)
        
        # Compute covariance matrices
        self.G1, self.G2 = self._compute_covariance_matrices(source_loader, target_loader)
        
        # Prepare validation data
        self.v1, self.v2 = self._prepare_validation_data(source_loader, target_loader)
        
        # Prepare anchor points
        self.anchor1, self.anchor2 = self._prepare_anchor_points(source_loader, target_loader, num_anchors)
        
        if self.anchor1 is None:
            if self.verbose:
                print("No anchor points used")
            self.supervised_w = 0.0
        else:
            if self.verbose:
                print(f"Using {self.anchor1.shape[0]} anchor points")
        
        # Create data iterators
        train_source_iter = ForeverDataIterator(source_loader)
        train_target_iter = ForeverDataIterator(target_loader)
        
        # Calculate iterations per epoch
        iters_per_epoch = max(len(source_loader), len(target_loader))
        
        # Training loop
        z_update_count = 0
        batches_done = 0
        
        if self.verbose:
            print(f"Starting training for {n_epochs} epochs...")
        
        for epoch in trange(n_epochs, desc="Training Epochs", disable=not self.verbose):
            noise_factor = 1.0 - (epoch / n_epochs)
            
            ep_cca_losses = []
            ep_f_losses = []
            ep_z_losses = []
            
            for batch_no in range(iters_per_epoch):
                # Get batch data
                X1, Y1 = next(train_source_iter)
                X2, Y2 = next(train_target_iter)
                X1, X2 = X1.to(self.device), X2.to(self.device)
                
                # Ensure same batch size
                min_samples = min(X1.shape[0], X2.shape[0])
                X1 = X1[:min_samples]
                X2 = X2[:min_samples]
                
                # Label smoothing
                labels_true = (torch.ones((min_samples, 1)) - 
                             self.lsmooth * (torch.rand((min_samples, 1)) * 0.2 * noise_factor)).to(self.device)
                labels_false = (torch.zeros((min_samples, 1)) + 
                              self.lsmooth * (torch.rand((min_samples, 1)) * 0.2 * noise_factor)).to(self.device)
                
                # Initialize discriminator loss
                dist_f_loss = torch.tensor(0.0, device=self.device)
                
                # Discriminator update
                if z_update_count == self.n_z:
                    z_update_count = 0
                    
                    self.optimizer_f.zero_grad()
                    c1 = torch.matmul(self.Z1, X1.T).T
                    c2 = torch.matmul(self.Z2, X2.T).T
                    
                    dist_f_loss = (self.loss_func(self.f(c1), labels_true) + 
                                  self.loss_func(self.f(c2), labels_false))
                    dist_f_loss.backward()
                    self.optimizer_f.step()
                
                # Encoder update
                self.optimizer_z1.zero_grad()
                self.optimizer_z2.zero_grad()
                
                c1 = torch.matmul(self.Z1, X1.T).T
                c2 = torch.matmul(self.Z2, X2.T).T
                dist_z_loss = (self.loss_func(self.f(c1), labels_false) + 
                              self.loss_func(self.f(c2), labels_true))
                
                # Orthogonality losses
                or_loss1 = torch.tensor(0.0, device=self.device)
                or_loss2 = torch.tensor(0.0, device=self.device)
                
                if self.orthogonal_w > 0.0:
                    or_loss1 = (self.orthogonal_w * (1/self.D) * 
                               torch.norm(torch.matmul(torch.matmul(self.Z1, self.G1), self.Z1.T) - self.ID))
                    or_loss2 = (self.orthogonal_w * (1/self.D) * 
                               torch.norm(torch.matmul(torch.matmul(self.Z2, self.G2), self.Z2.T) - self.ID))
                
                # Supervised loss
                super_loss = torch.tensor(0.0, device=self.device)
                if self.supervised_w > 0.0 and self.anchor1 is not None:
                    c1_anchor = torch.matmul(self.Z1, self.anchor1.T).T
                    c2_anchor = torch.matmul(self.Z2, self.anchor2.T).T
                    super_loss = self.supervised_w * torch.mean((c1_anchor - c2_anchor)**2)
                
                # Total loss
                total_loss = dist_z_loss + or_loss1 + or_loss2 + super_loss
                total_loss.backward()
                
                self.optimizer_z1.step()
                self.optimizer_z2.step()
                z_update_count += 1
                
                # Compute validation losses
                with torch.no_grad():
                    est_c1 = torch.matmul(self.Z1, self.v1.T).T
                    est_c2 = torch.matmul(self.Z2, self.v2.T).T
                    cca_loss = torch.norm(est_c1 - est_c2, p='fro')
                
                ep_cca_losses.append(cca_loss.item())
                ep_f_losses.append(dist_f_loss.item())
                ep_z_losses.append(dist_z_loss.item())
                
                batches_done += 1
            
            # Store epoch losses
            self.training_history['cca_losses'].append(np.mean(ep_cca_losses))
            self.training_history['f_losses'].append(np.mean(ep_f_losses))
            self.training_history['z_losses'].append(np.mean(ep_z_losses))
            
            # Print progress
            if self.verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:3d}/{n_epochs}: "
                      f"CCA Loss: {self.training_history['cca_losses'][-1]:.4f}, "
                      f"F Loss: {self.training_history['f_losses'][-1]:.4f}, "
                      f"Z Loss: {self.training_history['z_losses'][-1]:.4f}")
        
        if self.verbose:
            print("Training completed!")
        
        return self
    
    def transform(self, X_source: Union[np.ndarray, torch.Tensor], 
                  X_target: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data to shared representation space.
        
        Parameters:
        -----------
        X_source : array-like, shape (n_samples, n_features)
            Source domain data
        X_target : array-like, shape (n_samples, n_features)
            Target domain data
            
        Returns:
        --------
        c1 : ndarray, shape (n_samples, D)
            Source domain shared representations
        c2 : ndarray, shape (n_samples, D)
            Target domain shared representations
        """
        
        if self.Z1 is None or self.Z2 is None:
            raise ValueError("Model must be fitted before calling transform")
        
        # Convert to tensors if needed
        if isinstance(X_source, np.ndarray):
            X_source = torch.FloatTensor(X_source)
        if isinstance(X_target, np.ndarray):
            X_target = torch.FloatTensor(X_target)
        
        X_source = X_source.to(self.device)
        X_target = X_target.to(self.device)
        
        with torch.no_grad():
            c1 = torch.matmul(self.Z1, X_source.T).T
            c2 = torch.matmul(self.Z2, X_target.T).T
        
        return c1.cpu().numpy(), c2.cpu().numpy()
    
    def fit_transform(self, X_source: Union[np.ndarray, torch.Tensor], 
                     X_target: Union[np.ndarray, torch.Tensor],
                     y_source: Optional[Union[np.ndarray, torch.Tensor]] = None,
                     y_target: Optional[Union[np.ndarray, torch.Tensor]] = None,
                     num_anchors: int = 256,
                     n_epochs: int = 76) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the model and transform the data.
        
        Parameters:
        -----------
        X_source : array-like, shape (n_samples, n_features)
            Source domain data
        X_target : array-like, shape (n_samples, n_features)
            Target domain data
        y_source : array-like, shape (n_samples,), optional
            Source domain labels
        y_target : array-like, shape (n_samples,), optional
            Target domain labels
        num_anchors : int, default=256
            Number of anchor points for supervision
        n_epochs : int, default=76
            Number of training epochs
            
        Returns:
        --------
        c1 : ndarray, shape (n_samples, D)
            Source domain shared representations
        c2 : ndarray, shape (n_samples, D)
            Target domain shared representations
        """
        
        self.fit(X_source, X_target, y_source, y_target, num_anchors, n_epochs)
        return self.transform(X_source, X_target)
    
    def get_training_history(self) -> dict:
        """Get training history."""
        return self.training_history.copy()
    
    def evaluate_knn_accuracy(self, c1: np.ndarray, c2: np.ndarray, k: int = 5) -> dict:
        """
        Evaluate k-NN accuracy between shared representations.
        
        Parameters:
        -----------
        c1 : ndarray, shape (n_samples, D)
            Source domain shared representations
        c2 : ndarray, shape (n_samples, D)
            Target domain shared representations
        k : int, default=5
            Number of nearest neighbors
            
        Returns:
        --------
        accuracy : dict
            Dictionary containing accuracy metrics
        """
        
        # Simple k-NN accuracy calculation
        nn_12 = NearestNeighbors(n_neighbors=k, metric='l1')
        nn_12.fit(c1)
        distances_12, indices_12 = nn_12.kneighbors(c2)
        
        nn_21 = NearestNeighbors(n_neighbors=k, metric='l1')
        nn_21.fit(c2)
        distances_21, indices_21 = nn_21.kneighbors(c1)
        
        # Calculate accuracy as percentage of points that are mutual nearest neighbors
        mutual_nn = 0
        for i in range(len(c1)):
            if i in indices_12[i]:
                mutual_nn += 1
        
        accuracy_12 = mutual_nn / len(c1)
        
        mutual_nn = 0
        for i in range(len(c2)):
            if i in indices_21[i]:
                mutual_nn += 1
        
        accuracy_21 = mutual_nn / len(c2)
        
        return {
            'source_to_target_accuracy': accuracy_12,
            'target_to_source_accuracy': accuracy_21,
            'average_accuracy': (accuracy_12 + accuracy_21) / 2.0
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.Z1 is None or self.Z2 is None:
            raise ValueError("Model must be fitted before saving")
        
        torch.save({
            'Z1': self.Z1,
            'Z2': self.Z2,
            'f_state_dict': self.f.state_dict(),
            'G1': self.G1,
            'G2': self.G2,
            'D': self.D,
            'orthogonal_w': self.orthogonal_w,
            'supervised_w': self.supervised_w,
            'training_history': self.training_history
        }, filepath)
        
        if self.verbose:
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.Z1 = checkpoint['Z1'].to(self.device)
        self.Z2 = checkpoint['Z2'].to(self.device)
        self.D = checkpoint['D']
        self.orthogonal_w = checkpoint['orthogonal_w']
        self.supervised_w = checkpoint['supervised_w']
        self.training_history = checkpoint['training_history']
        
        # Reinitialize discriminator
        self.f = Discriminator(self.D, 1).to(self.device)
        self.f.load_state_dict(checkpoint['f_state_dict'])
        
        # Load covariance matrices if available
        if 'G1' in checkpoint:
            self.G1 = checkpoint['G1'].to(self.device)
        if 'G2' in checkpoint:
            self.G2 = checkpoint['G2'].to(self.device)
        
        if self.verbose:
            print(f"Model loaded from {filepath}")
