import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

from data_load import *
from graph_vae import GraphVAE
# %%
# def recreate(z, in_shape):
#     z.shape = (batch, 1000, 3)
#     x = torch.zeros(in_shape)    

def loss_fn(net, x):
    # What terms to include ?
    # x.shape = (batch, 125, 125, 3)
    X, A = preprocess(x)
    Y, A2, mu, logvar, L1, L2 = net(X, A)
    mse = torch.nn.MSELoss()
    return mse(X, Y) + L1 + L2
        

def train_loop(net: GraphVAE, epochs, batch_size, lr=1e-3):
    dataset = Train_Dataset(batch_size)
    data_loader = torch.utils.data.DataLoader(dataset)
    opt = torch.optim.Adam(net.parameters(), lr)

    for ep in range(epochs):
        for (x, m0, pt, y) in data_loader:
            opt.zero_grad()
            loss = loss_fn(net, x)
            loss.backward()
            
            optimizer.step()
    dataset.close()
