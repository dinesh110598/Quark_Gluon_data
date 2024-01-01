import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

def deg_norm(adj):
    EPS = 1e-15
    d_inv_sqrt = 1/(torch.sum(adj, -1) + EPS).sqrt()
    return d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)

def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

    return out

def dense_mincut_pool(x, adj, s):

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    k = s.size(-1)

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    
    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    # Degree normalization
    out_adj = deg_norm(out_adj)

    return out, out_adj, mincut_loss, ortho_loss

class MinCut_Pool(nn.Module):
    def __init__(self, in_channels, n_clusters):
        super().__init__()
        # self.linear = nn.Linear(in_channels, n_clusters)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, n_clusters)
        )
        
    def forward(self, X, A, mask=None):
        S = self.mlp(X)
        # Processing S directly is useful for GraphVAE
        S = torch.softmax(S, -1)
        if mask is not None:
            mask = mask.view(X.shape[0], X.shape[1], 1)
            X = X * mask
            S = S * mask
        
        return (S,) + dense_mincut_pool(X, A, S)
    
class GraphVAE(nn.Module):
    def __init__(self, max_nodes=1000, in_channels=4, 
                 hidden_channels=32, latent_dim=32):
        super().__init__()
        assert max_nodes%20 == 0
        
        hidden = hidden_channels
        self.latent_dim = latent_dim
        
        self.sage = nn.ModuleList([
            gnn.DenseSAGEConv(in_channels, 32, normalize=False),
            gnn.DenseSAGEConv(32, 64, normalize=False),
            gnn.DenseSAGEConv(64, 64, normalize=False)
        ])
        # self.lin = nn.ModuleList([
        #     nn.Linear(16, 16),
        #     nn.Linear(64, 64)
        # ])
        self.drop = nn.ModuleList([
            nn.Dropout(0.5),
            nn.Dropout(0.4),
            nn.Dropout(0.3)
        ])
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm1d(max_nodes),
            nn.BatchNorm1d(max_nodes//4),
            nn.BatchNorm1d(max_nodes//20)
        ])
        self.pool = nn.ModuleList([
            MinCut_Pool(32, max_nodes//4),
            MinCut_Pool(64, max_nodes//20)
        ])
        
        self.tr_z = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2*latent_dim))
        self.tr_rev = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64))
        
        self.revsage = nn.ModuleList([
            gnn.DenseSAGEConv(32, in_channels, normalize=False),
            gnn.DenseSAGEConv(64, 32, normalize=False),
            gnn.DenseSAGEConv(64, 64, normalize=False)
        ])
        # self.rev_lin = nn.ModuleList([
        #     nn.Linear(16, 16),
        #     nn.Linear(64, 64)
        # ])
        
    def upsample(self, X, A, S):
        X = torch.bmm(S, X)
        A = torch.bmm(S, torch.bmm(A, S.permute((0,2,1))))
        return X, A
        
    def encode(self, X, A, mask):
        pool_S = ()
        mincut_loss = 0.
        ortho_loss = 0.
        
        for i in range(3):
            X = F.relu(self.sage[i](X, A, mask if i==0 else None))
            # if i<2:
            #     X = F.relu(self.lin[i](X))
            X = self.batch_norm[i](X)
            
            if i<2:
                X = self.drop[i](X)
                S, X, A, mc, on = self.pool[i](X, A, mask if i==0 else None)
                
                pool_S += (S,)
                mincut_loss += mc
                ortho_loss += on
        
        mu, sig = torch.chunk(self.tr_z(X), 2, -1)
        return (mu, sig, A, 
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
                A = deg_norm(A)
                # Z = F.relu(self.rev_lin[i-1](Z))
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