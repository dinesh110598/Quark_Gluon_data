import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

class MinCut_Pool(nn.Module):
    def __init__(self, in_channels, n_clusters):
        super().__init__()
        self.linear = nn.Linear(in_channels, n_clusters)
        
    def forward(self, X, A):
        S = self.linear(X)
        return (S,) + gnn.dense_mincut_pool(X, A, S)
    
class GraphVAE(nn.Module):
    def __init__(self, hidden_channels=32, latent_dim=8):
        super().__init__()
        # Default values
        # max_nodes = 1000
        # depth = 6
        # latent_dim = 16
        
        hidden = hidden_channels
        self.latent_dim = latent_dim
        
        self.sage = nn.ModuleList([
            gnn.DenseSAGEConv(3, hidden),
            gnn.DenseSAGEConv(hidden, hidden),
            gnn.DenseSAGEConv(hidden, hidden)
        ])
        self.drop = nn.ModuleList([
            nn.Dropout(0.5),
            nn.Dropout(0.4),
            nn.Dropout(0.3)
        ])
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm1d(1000),
            nn.BatchNorm1d(500),
            nn.BatchNorm1d(250)
        ])
        self.pool = nn.ModuleList([
            MinCut_Pool(hidden, 500),
            MinCut_Pool(hidden, 250)
        ])
        
        self.tr_z = nn.Linear(hidden, latent_dim)
        
        self.tr_rev = nn.Linear(latent_dim, hidden)
        self.revsage = nn.ModuleList([
            gnn.DenseSAGEConv(hidden, 2*hidden),
            gnn.DenseSAGEConv(hidden, hidden),
            gnn.DenseSAGEConv(hidden, hidden)
        ])
        
        self.out_hits = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 2),
            nn.Sigmoid()
        )
        self.out_ener = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1)
        )
        
    def upsample(self, X, A, S):
        X = torch.bmm(S, X)
        A = torch.bmm(S, torch.bmm(A, S.permute((0,2,1))))
        return X, A
        
    def encode(self, X, A):
        pool_S = ()
        mincut_loss = 0.
        ortho_loss = 0.
        
        for i in range(3):
            X = F.relu(self.sage[i](X, A))
            X = self.batch_norm[i](X)
            X = self.drop[i](X)
            if i<2:
                S, X, A, mc, on = self.pool[i](X, A)
                pool_S += (S,)
                mincut_loss += mc
                ortho_loss += on
        
        return (self.tr_z(X), A, 
                pool_S, mincut_loss, ortho_loss)
    
    def decode(self, Z, A, pool_S):
        Z = F.leaky_relu(self.tr_rev(Z))
        Z = self.drop[-1](Z)
        
        for i in reversed(range(3)):
            Z = F.relu(self.revsage[i](Z, A))
            if i!=0:
                Z = self.batch_norm[i](Z)
                Z = self.drop[i](Z)
                # Need to check if upsample really works
                Z, A = self.upsample(Z, A, pool_S[i-1])
                Z = F.leaky_relu(Z)
                A = F.sigmoid(A)
                
        Z1, Z2 = torch.split(Z, 2, -1)
        X_hits = self.out_hits(Z1)
        X_ener = self.out_ener(Z2)
        return torch.cat((X_hits, X_ener), -1), A
    
    def forward(self, X, A):
        Z, A, pool_S, L1, L2 = self.encode(X, A)
        X, A = self.decode(Z, A, pool_S)
        return X, A, L1, L2