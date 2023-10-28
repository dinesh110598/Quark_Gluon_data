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
    def __init__(self, in_channels, hidden, out_channels):
        super().__init__()
        # Default values
        # max_nodes = 1000
        # depth = 6
        # latent_dim = 16
        
        self.hidden = hidden
        self.out_channels = out_channels
        
        self.sage = nn.ModuleList([
            gnn.DenseSAGEConv(in_channels, hidden),
            gnn.DenseSAGEConv(hidden, hidden),
            gnn.DenseSAGEConv(hidden, out_channels)
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
        
        self.tr_mu = nn.Linear(out_channels, 16)
        self.tr_var = nn.Linear(out_channels, 16)
        
        self.tr_rev = nn.Linear(16, out_channels)
        self.revsage = nn.ModuleList([
            gnn.DenseSAGEConv(hidden, in_channels),
            gnn.DenseSAGEConv(hidden, hidden),
            gnn.DenseSAGEConv(out_channels, hidden)
        ])
        
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
        
        return (self.tr_mu(X), self.tr_var(X), A, 
                pool_S, mincut_loss, ortho_loss)
    
    def reparameterize(self, mu, logvar):
        sig = torch.exp(0.5*logvar)
        eps = torch.randn_like(sig)
        return mu + eps*sig
    
    def decode(self, Z, A, pool_S):
        Z = F.leaky_relu(self.tr_rev(Z))
        out1 = self.drop[-1](Z)
        
        for i in reversed(range(3)):
            Z = F.relu(self.revsage[i](Z, A))
            if i!=0:
                Z = self.batch_norm[i](Z)
                Z = self.drop[i](Z)
                
                Z, A = self.upsample(Z, A, pool_S[i-1])
                Z = F.leaky_relu(Z)
                A = F.sigmoid(A)
                
        return Z, A
    
    def forward(self, X, A):
        mu, logvar, A, pool_S, L1, L2 = self.encode(X, A)
        Z = self.reparameterize(mu, logvar)
        X, A = self.decode(Z, A, pool_S)
        return X, A, mu, logvar, L1, L2