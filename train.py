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
    X, A, mask = preprocess(x)
    # Reconstructed nodes and edges
    Y, A2, L1, L2 = net(X, A)
    mse = torch.nn.MSELoss()
    
    return mse(X, Y) + L1 + L2

def train_loop(net: GraphVAE, epochs, batch_size, lr=1e-3):
    dataset = get_train_dataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size)
    opt = torch.optim.Adam(net.parameters(), lr)

    for ep in range(epochs):
        for step, (x,) in enumerate(data_loader):
            opt.zero_grad()
            loss = loss_fn(net, x)
            with torch.no_grad():
                print(step, loss)
            loss.backward()
            
            if (step+1)%100 == 0:
                torch.save(net.state_dict(), 
                           "Saves/Checkpoints/s_{}.pth".format(step+1))
            opt.step()
            
def loss_infer(net, x):
    """
    Inference loss function
    """
    # Graph nodes and edges
    X, A, counts = preprocess(x)
    # Reconstructed nodes and edges
    Y, A2, L1, L2 = net(X, A)
    
    # Convert back to image
    ecal = reconstruct_img(Y, counts)
    
    mse = torch.nn.MSELoss()
    
    return mse(X, Y), ecal, counts
    
# %%
# net = GraphVAE(3, 32, 8)
# net.load_state_dict(torch.load("Saves/L_50k_2.pth"))
# dataset = get_train_dataset(10_000)
# dataloader = torch.utils.data.DataLoader(dataset, 200, True)
# for (x,) in dataloader:
#     img1 = x[:, :, :, 1]
#     with torch.no_grad():
#         L, img2, counts = loss_infer(net, x)
#     break
# %%