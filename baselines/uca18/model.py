import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, linear=False):
        super().__init__()
        if linear:
            self.net = nn.Linear(input_dim, latent_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, linear=False):
        super().__init__()
        if linear:
            self.net = nn.Linear(latent_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)

class UCA18:
    """
    Unsupervised Correlation Analysis (UCA) Baseline.
    Based on Hoshen & Wolf (arXiv:1804.00347).
    """
    def __init__(self, input_dim1, input_dim2, latent_dim, 
                 lr=1e-4, lambda_rec=10.0, lambda_cyc=10.0, lambda_ortho=1.0,
                 batch_size=64, device='cuda' if torch.cuda.is_available() else 'cpu',
                 linear=False):
        self.latent_dim = latent_dim
        self.device = device
        self.batch_size = batch_size
        self.lambda_rec = lambda_rec
        self.lambda_cyc = lambda_cyc
        self.lambda_ortho = lambda_ortho

        self.E1 = Encoder(input_dim1, latent_dim, linear=linear).to(device)
        self.E2 = Encoder(input_dim2, latent_dim, linear=linear).to(device)
        self.D1 = Decoder(latent_dim, input_dim1, linear=linear).to(device)
        self.D2 = Decoder(latent_dim, input_dim2, linear=linear).to(device)
        self.C = Discriminator(latent_dim).to(device)

        self.opt_G = torch.optim.Adam(
            list(self.E1.parameters()) + list(self.E2.parameters()) + 
            list(self.D1.parameters()) + list(self.D2.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )
        self.opt_C = torch.optim.Adam(self.C.parameters(), lr=lr, betas=(0.5, 0.999))

    def _ortho_loss(self, E, X):
        z = E(X)
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (z.shape[0] - 1)
        return torch.norm(cov - torch.eye(self.latent_dim).to(self.device))

    def fit(self, X1, X2, n_epochs=100, verbose=True):
        X1 = torch.tensor(X1, dtype=torch.float32).to(self.device)
        X2 = torch.tensor(X2, dtype=torch.float32).to(self.device)
        
        dataset1 = TensorDataset(X1)
        dataset2 = TensorDataset(X2)
        loader1 = DataLoader(dataset1, batch_size=self.batch_size, shuffle=True)
        loader2 = DataLoader(dataset2, batch_size=self.batch_size, shuffle=True)

        for epoch in range(n_epochs):
            g_losses, c_losses = [], []
            for (x1,), (x2,) in zip(loader1, loader2):
                x1, x2 = x1.to(self.device), x2.to(self.device)

                # --- Train Discriminator ---
                self.opt_C.zero_grad()
                z1 = self.E1(x1).detach()
                z2 = self.E2(x2).detach()
                
                loss_C = F.binary_cross_entropy(self.C(z1), torch.ones_like(self.C(z1)) * 0.9) + \
                         F.binary_cross_entropy(self.C(z2), torch.zeros_like(self.C(z2)) * 0.1)
                loss_C.backward()
                self.opt_C.step()
                c_losses.append(loss_C.item())

                # --- Train Generators (Encoders + Decoders) ---
                self.opt_G.zero_grad()
                z1 = self.E1(x1)
                z2 = self.E2(x2)

                # Adversarial loss
                loss_adv = F.binary_cross_entropy(self.C(z1), torch.zeros_like(self.C(z1))) + \
                           F.binary_cross_entropy(self.C(z2), torch.ones_like(self.C(z2)))

                # Reconstruction loss
                loss_rec = F.mse_loss(self.D1(z1), x1) + F.mse_loss(self.D2(z2), x2)

                # Cycle loss
                # X1 -> Z1 -> X2_est -> Z2_est -> X1_back
                x2_est = self.D2(z1)
                z2_est = self.E2(x2_est)
                x1_back = self.D1(z2_est)
                
                x1_est = self.D1(z2)
                z1_est = self.E1(x1_est)
                x2_back = self.D2(z1_est)
                
                loss_cyc = F.mse_loss(x1_back, x1) + F.mse_loss(x2_back, x2)

                # Orthogonality loss
                loss_ortho = self._ortho_loss(self.E1, x1) + self._ortho_loss(self.E2, x2)

                loss_G = loss_adv + self.lambda_rec * loss_rec + self.lambda_cyc * loss_cyc + self.lambda_ortho * loss_ortho
                loss_G.backward()
                self.opt_G.step()
                g_losses.append(loss_G.item())

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:3d}/{n_epochs} | G Loss: {np.mean(g_losses):.4f} | C Loss: {np.mean(c_losses):.4f}")

    def transform(self, X1, X2):
        self.E1.eval()
        self.E2.eval()
        with torch.no_grad():
            X1_t = torch.tensor(X1, dtype=torch.float32).to(self.device)
            X2_t = torch.tensor(X2, dtype=torch.float32).to(self.device)
            z1 = self.E1(X1_t).cpu().numpy()
            z2 = self.E2(X2_t).cpu().numpy()
        return z1, z2
