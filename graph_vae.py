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
    
class ForwardSAGEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.3):
        super().__init__()

        self.inp_c = in_channels
        self.out_c = out_channels
        
        self.sage1 = gnn.DenseSAGEConv(in_channels, out_channels)
        self.lin = nn.Linear(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(drop)
        self.sage2 = gnn.DenseSAGEConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, X, A, mask=None):
        X2 = F.relu(self.sage1(X, A, mask))
        X2 = self.lin(X2)
        X2 = self.bn1(X2)
        X2 = self.drop(X2)
        X2 = F.relu(self.sage2(X2, A))
        X2 = self.bn2(X2)

        return X2 + X
    
class ReverseSAGEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop):
        super().__init__()

        self.inp_c = in_channels
        self.out_c = out_channels

        self.sage2 = gnn.DenseSAGEConv(in_channels, in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.drop = nn.Dropout(drop)
        self.lin = nn.Linear(in_channels, in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.sage1 = gnn.DenseSAGEConv(in_channels, out_channels)

    def forward(self, X, A):
        X2 = F.relu(self.sage2(X, A))
        X2 = self.bn2(X2)
        X2 = self.drop(X2)
        X2 = self.lin(X2)
        X2 = F.relu(self.sage1(X, A))
        X2 = self.bn1(X2)

        return X2 + X
    
class GraphVAE(nn.Module):
    def __init__(self, max_nodes=1000, in_channels=6, 
                 hidden_channels=32, latent_dim=32):
        super().__init__()
        assert max_nodes%20 == 0
        
        hidden = hidden_channels
        self.latent_dim = latent_dim

        self.forward_blocks = nn.ModuleList([
            ForwardSAGEBlock(in_channels, hidden),
            ForwardSAGEBlock(hidden, 2*hidden),
            ForwardSAGEBlock(2*hidden, 2*hidden)
        ])
        
        self.pool = nn.ModuleList([
            MinCut_Pool(hidden, max_nodes//4),
            MinCut_Pool(2*hidden, max_nodes//20)
        ])
        
        self.tr_z = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2*latent_dim))
        self.tr_rev = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2*hidden))
        
        self.reverse_blocks = nn.ModuleList([
            ReverseSAGEBlock(2*hidden, 2*hidden),
            ReverseSAGEBlock(2*hidden, hidden),
            ReverseSAGEBlock(hidden, in_channels)
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
            X = self.forward_blocks[i](X, A, mask if i==0 else None)
            
            if i<2:
                S, X, A, mc, on = self.pool[i](X, A, mask if i==0 else None)
                pool_S += (S,)
                mincut_loss += mc
                ortho_loss += on
        
        mu, sig = torch.chunk(self.tr_z(X), 2, -1)
        return (mu, sig, A, 
                pool_S, mincut_loss, ortho_loss)
    
    def decode(self, Z, A, pool_S):
        Z = self.reverse_blocks[i](Z, A)
        
        for i in reversed(range(3)):
            Z = F.relu(self.revsage[i](Z, A))
            if i>0:
                # Need to check if upsample really works
                Z, A = self.upsample(Z, A, pool_S[i-1])
                Z = F.leaky_relu(Z)
                A = deg_norm(A)
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