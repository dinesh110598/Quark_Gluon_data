import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

from data_load import *
from graph_vae import GraphVAE
# %%
def loss_fn(net, x, device):
    """
    Loss function with simple mean-square distance between
    autoencoder inputs and outputs, along with spatial graph
    pooling losses and KL-divergence of the latent random
    variables wrt N(0,1)
    """
    # What terms to include ?
    # x.shape = (batch, 125, 125, 3)
    
    # Graph nodes and edges
    X, A, mask, _ = preprocess(x, device)
    # Reconstructed nodes and edges
    Y, A2, mu, logvar, L1, L2 = net(X, A, mask)
    
    mse_hit = torch.nn.MSELoss()(X[:, :, :2], Y[:, :, :2])
    mse_ener = torch.nn.MSELoss()(X[:, :, 2], Y[:, :, 2])
    mse = mse_hit + 50*mse_ener
    KL_div = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    graph_reg = L1 + L2
    # wt_reg = sum([p.abs().sum() for p in net.parameters()])
    
    return mse + KL_div + graph_reg, mse_hit, mse_ener

def train_loop(net: GraphVAE, epochs, batch_size, lr=1e-3, 
               device=torch.device("cpu")):
    dataset = get_train_dataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size)
    opt = torch.optim.Adam(net.parameters(), lr)

    for ep in range(epochs):
        ep_loss, ep_E_mse, ep_hit_mse = 0., 0., 0.
        for i, (x,) in enumerate(data_loader):
            x = x.to(device)
            opt.zero_grad()
            loss, hit_mse, E_mse = loss_fn(net, x, device)
            loss.backward()
            
            ep_loss += float(loss.item())
            ep_E_mse += float(E_mse.item())
            ep_hit_mse += float(hit_mse.item())
            
            opt.step()
            break
            
        torch.save(net.state_dict(), 
                   "Saves/Checkpoints/ep_{}.pth".format(ep+1))
            
        print("Epoch : {}".format(ep+1), 
              "Loss: {:.4f}".format(ep_loss/250.),
              "E mse: {:.4f}".format(ep_E_mse/250.), 
              "Hit mse: {:.4f}".format(ep_hit_mse/250.))
            
def loss_infer(net, x):
    """
    Inference loss function
    """
    # Graph nodes and edges
    X, A, mask, counts = preprocess(x)
    # Reconstructed nodes and edges
    Y, A2, mu, logvar, L1, L2 = net(X, A, mask)
    
    # Convert back to image
    ecal = reconstruct_img(Y, counts)
    
    mse = torch.nn.MSELoss()
    
    return mse(X, Y), ecal, counts
    
# %%
# net = GraphVAE()
# net.load_state_dict(torch.load("Saves/Checkpoints/ep_15.pth"))
# dataset = get_train_dataset(400)
# dataloader = torch.utils.data.DataLoader(dataset, 200, True)
# for (x,) in dataloader:
#     img1 = x[:, :, :, 1]
#     with torch.no_grad():
#         L, img2, counts = loss_infer(net, x)
#     break
# %%