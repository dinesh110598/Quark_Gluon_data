import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

from data_load import *
from graph_vae import GraphVAE
# %%
# def loss_fn(net, x, device):
#     """
#     Loss function with simple mean-square distance between
#     autoencoder inputs and outputs, along with spatial graph
#     pooling losses and KL-divergence of the latent random
#     variables wrt N(0,1)
#     """
#     # What terms to include ?
#     # x.shape = (batch, 125, 125, 3)
    
#     # Graph nodes and edges
#     X, A, mask, _ = preprocess(x, device)
#     # Reconstructed nodes and edges
#     Y, A2, mu, logvar, L1, L2 = net(X, A, mask)
    
#     mse_hit = torch.nn.MSELoss()(X[:, :, :2], Y[:, :, :2])
#     mse_ener = torch.nn.MSELoss()(X[:, :, 2], Y[:, :, 2])
#     mse = mse_hit + mse_ener
#     KL_div = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     graph_reg = L1 + L2
#     # wt_reg = sum([p.abs().sum() for p in net.parameters()])
    
#     return mse + KL_div + graph_reg, mse_hit, mse_ener

# def train_loop(net: GraphVAE, epochs, batch_size, lr=1e-3, 
#                device=torch.device("cpu")):
#     dataset = get_train_dataset()
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size)
#     opt = torch.optim.Adam(net.parameters(), lr)

#     for ep in range(epochs):
#         ep_loss, ep_E_mse, ep_hit_mse = 0., 0., 0.
#         for i, (x,) in enumerate(data_loader):
#             x = x.to(device)
#             opt.zero_grad()
#             loss, hit_mse, E_mse = loss_fn(net, x, device)
#             loss.backward()
            
#             ep_loss += float(loss.item())
#             ep_E_mse += float(E_mse.item())
#             ep_hit_mse += float(hit_mse.item())
            
#             opt.step()
            
#         torch.save(net.to("cpu").state_dict(), 
#                    "Saves/Checkpoints/ep_{}.pth".format(ep+1))
            
#         print("Epoch : {}".format(ep+1), 
#               "Loss: {:.4f}".format(ep_loss/250.),
#               "E mse: {:.4f}".format(ep_E_mse/250.), 
#               "Hit mse: {:.4f}".format(ep_hit_mse/250.))
# %%
def loss_fn(net, X, A, mask, periodic=True):
    # Reconstructed nodes and edges
    Y, A2, mu, logvar, L1, L2 = net(X, A, mask)
    X = X * mask.unsqueeze(-1)
    Y = Y * mask.unsqueeze(-1)
    
    if periodic:
        # Make xhits compatible with periodic boundary
        with torch.no_grad():
            X_xhit = X[:, :, 0]
            Y_xhit = Y[:, :, 0]
            flag1 = (X_xhit - Y_xhit).abs() > 0.5
            flag2 = X_xhit < 0.5
            
            X_xhit[torch.logical_and(flag1, flag2)] += 1.
            X_xhit[torch.logical_and(
                flag1, torch.logical_not(flag2))] -= 1.
    else:
        X_xhit = X[:, :, 0]
    
    mse_xhit = torch.nn.MSELoss()(X_xhit, Y[:, :, 0])
    mse_yhit = torch.nn.MSELoss()(X[:, :, 1], Y[:, :, 1])
    mse_hit = mse_xhit + mse_yhit
    
    mse_ener = torch.nn.MSELoss()(X[:, :, 2], Y[:, :, 2])
    # mse_chan = torch.nn.MSELoss()(X[:, :, 3], Y[:, :, 3])
    mse = mse_hit + mse_ener # + mse_chan
    KL_div = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    graph_reg = L1 + L2
    
    # wt_reg = sum([p.abs().sum() for p in net.parameters()])
    
    return mse + KL_div + 0.1*graph_reg, mse_hit, mse_ener, L2

def train_loop(net: GraphVAE, epochs, batch_size=64, lr=1e-3, 
                device=torch.device("cpu"), periodic=True):
    net.train()
    dataset = get_train_dataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, True)
    opt = torch.optim.Adam(net.parameters(), lr)

    loss_list = []
    E_mse_list = []
    hit_mse_list = []
    for ep in range(epochs):
        net = net.to(device)
        ep_loss, ep_E_mse, ep_hit_mse, ep_L2, count = 0., 0., 0., 0., 0.
        for (X, NL, mask) in data_loader:
            X = X.to(device)
            mask = mask.to(device)
            A = eval_A(NL.to(device).int())*mask.unsqueeze(2)
            
            opt.zero_grad()
            loss, hit_mse, E_mse, L2 = loss_fn(net, X, A, mask, periodic)
            loss.backward()
            
            ep_loss += float(loss.item())
            ep_E_mse += float(E_mse.item())
            ep_hit_mse += float(hit_mse.item())
            ep_L2 += float(L2.item())
            count += 1
            
            opt.step()
            
        torch.save(net.to("cpu").state_dict(), 
                   "Saves/Checkpoints/ep_{}.pth".format(ep+1))
        
        ep_loss /= count
        ep_hit_mse /= count
        ep_E_mse /= count
        ep_L2 /= count
        print("Epoch : {}".format(ep+1), 
              "Loss: {:.5f}".format(ep_loss),
              "E mse: {:.5f}".format(ep_E_mse), 
              "Hit mse: {:.5f}".format(ep_hit_mse),
              "L2: {:.5f}".format(ep_L2))
        
        loss_list.append(ep_loss)
        E_mse_list.append(ep_E_mse)
        hit_mse_list.append(ep_hit_mse)
    
    net.eval()
    return loss_list, E_mse_list, hit_mse_list
# %%
# net = GraphVAE(400, 3)
# device = torch.device("cuda:0")
# loss2, E_mse2, hit_mse2 = train_loop(net, 1, 250, 5e-4, device, True)
# %%