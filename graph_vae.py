import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

class MinCut_Pool(nn.Module):
    def __init__(self, in_channels, n_clusters):
        super().__init__()
        self.linear = nn.Linear(in_channels, n_clusters)
        
    def forward(self, X, A, mask=None):
        S = self.linear(X)
        return (S,) + gnn.dense_mincut_pool(X, A, S, mask)
    
class GraphVAE(nn.Module):
    def __init__(self, hidden_channels=32, latent_dim=8):
        super().__init__()
        # Default values
        # max_nodes = 300
        # depth = 6
        # latent_dim = 16
        
        hidden = hidden_channels
        self.latent_dim = latent_dim
        
        self.sage = nn.ModuleList([
            gnn.DenseSAGEConv(3, hidden, normalize=True),
            gnn.DenseSAGEConv(hidden, hidden//2, normalize=True),
            gnn.DenseSAGEConv(hidden//2, hidden//4, normalize=True)
        ])
        self.drop = nn.ModuleList([
            nn.Dropout(0.5),
            nn.Dropout(0.4),
            nn.Dropout(0.3)
        ])
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm1d(300),
            nn.BatchNorm1d(75),
            nn.BatchNorm1d(15)
        ])
        self.pool = nn.ModuleList([
            MinCut_Pool(hidden, 75),
            MinCut_Pool(hidden//2, 15)
        ])
        
        self.tr_z = nn.Linear(hidden//4, latent_dim)
        self.tr_rev = nn.Linear(latent_dim, hidden//4)
        
        self.revsage = nn.ModuleList([
            gnn.DenseSAGEConv(hidden, 3, normalize=False),
            gnn.DenseSAGEConv(hidden//2, hidden, normalize=False),
            gnn.DenseSAGEConv(hidden//4, hidden//2, normalize=False)
        ])
        
    def upsample(self, X, A, S):
        X = torch.bmm(S, X)
        A = torch.bmm(S, torch.bmm(A, S.permute((0,2,1))))
        return X, A
        
    def encode(self, X, A, mask):
        pool_S = ()
        mincut_loss = 0.
        ortho_loss = 0.
        
        for i in range(3):
            X = F.relu(self.sage[i](X, A))
            X = self.batch_norm[i](X)
            X = self.drop[i](X)
            if i<2:
                S, X, A, mc, on = self.pool[i](X, A, mask if i==0 else None)
                pool_S += (S,)
                mincut_loss += mc
                ortho_loss += on
        
        return (self.tr_z(X), self.tr_z(X), A, 
                pool_S, mincut_loss, ortho_loss)
    
    def decode(self, Z, A, pool_S):
        Z = F.leaky_relu(self.tr_rev(Z))
        Z = self.drop[-1](Z)
        
        for i in reversed(range(3)):
            Z = F.relu(self.revsage[i](Z, A))
            if i>0:
                Z = self.batch_norm[i](Z)
                Z = self.drop[i](Z)
                # Need to check if upsample really works
                Z, A = self.upsample(Z, A, pool_S[i-1])
                Z = F.leaky_relu(Z)
                A = F.sigmoid(A)
                
        return Z, A
    
    def sample_z(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, X, A, mask):
        Z_mu, Z_logvar, A, pool_S, L1, L2 = self.encode(X, A, mask)
        Z = self.sample_z(Z_mu, Z_logvar)
        X, A = self.decode(Z, A, pool_S)
        return X, A, Z_mu, Z_logvar, L1, L2