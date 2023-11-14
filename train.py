import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

from data_load import *
from graph_vae import GraphVAE
# %%
def loss_fn(net, x):
    """
    Loss function with simple mean-square distance between
    autoencoder inputs and outputs, along with spatial graph
    pooling losses and KL-divergence of the latent random
    variables wrt N(0,1)
    """
    # What terms to include ?
    # x.shape = (batch, 125, 125, 3)
    
    # Graph nodes and edges
    X, A = preprocess(x)
    # Reconstructed nodes and edges
    Y, A2, mu, logvar, L1, L2 = net(X, A)
    mse = torch.nn.MSELoss()
    # KL-divergence b/w latent distribution and N(0,1)
    KL_div = 0.5*(logvar + (1 + mu**2)/logvar.exp())
    return mse(X, Y) + L1 + L2 + KL_div.mean()

def train_loop(net: GraphVAE, epochs, batch_size, lr=1e-3):
    dataset = get_train_dataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size)
    opt = torch.optim.Adam(net.parameters(), lr)

    for ep in range(epochs):
        for (x,) in data_loader:
            opt.zero_grad()
            loss = loss_fn(net, x)
            with torch.no_grad():
                print(loss)
            loss.backward()
            
            opt.step()
    dataset.close()
# %%
# net = GraphVAE(3, 32, 8)
# train_loop(net, 1, 50)
# %%