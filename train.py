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
            
        torch.save(net.to("cpu").state_dict(), 
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
def loss_fn2(net, X, A, mask):
    # Reconstructed nodes and edges
    Y, A2, mu, logvar, L1, L2 = net(X, A, mask)
    
    mse_hit = torch.nn.MSELoss()(X[:, :, :2], Y[:, :, :2])
    mse_ener = torch.nn.MSELoss()(X[:, :, 2], Y[:, :, 2])
    mse = mse_hit + mse_ener
    KL_div = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    graph_reg = L1 + L2
    # wt_reg = sum([p.abs().sum() for p in net.parameters()])
    
    return (mse + KL_div)/64. + graph_reg, mse_hit, mse_ener

def train_loop2(net: GraphVAE, epochs, batch_size=250, lr=1e-3, 
                device=torch.device("cpu")):
    dataset = Train_Dataset(batch_size)
    data_loader = torch.utils.data.DataLoader(dataset)
    opt = torch.optim.Adam(net.parameters(), lr)

    for ep in range(epochs):
        net = net.to(device)
        ep_loss, ep_E_mse, ep_hit_mse = 0., 0., 0.
        for (X, A, mask) in data_loader:
            X = X[0].to(device)
            A = A[0].to(device)
            mask = mask[0].to(device)
            
            opt.zero_grad()
            loss, hit_mse, E_mse = loss_fn2(net, X, A, mask)
            loss.backward()
            
            ep_loss += float(loss.item())
            ep_E_mse += float(E_mse.item())
            ep_hit_mse += float(hit_mse.item())
            
            opt.step()
            
        torch.save(net.to("cpu").state_dict(), 
                   "Saves/Checkpoints/ep_{}.pth".format(ep+1))
            
        print("Epoch : {}".format(ep+1), 
              "Loss: {:.4f}".format(ep_loss/200.),
              "E mse: {:.4f}".format(ep_E_mse/200.), 
              "Hit mse: {:.4f}".format(ep_hit_mse/200.))
    
    dataset.close()
    
def loss_infer2(net, X, A, mask):
    """
    Inference loss function
    """
    # Reconstructed nodes and edges
    Y, A2, mu, logvar, L1, L2 = net(X, A, mask)
    counts = torch.sum(mask, 1)
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