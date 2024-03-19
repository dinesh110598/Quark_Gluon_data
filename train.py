import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

from data_load import *
from graph_vae import GraphVAE
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
    
    mse_xhit = nn.MSELoss()(X_xhit, Y[:, :, 0])
    mse_yhit = nn.MSELoss()(X[:, :, 1], Y[:, :, 1])
    mse_hit = mse_xhit + mse_yhit
    
    mse_ener = nn.MSELoss()(X[:, :, 2], Y[:, :, 2])
    # CE loss with masking 
    mce_loss = nn.CrossEntropyLoss(reduction='none')
    X_chan = X[:, :, 3:].view(-1, 3)
    Y_chan = Y[:, :, 3:].view(-1, 3)
    mce_chan = mce_loss(Y_chan, X_chan)
    mce_chan = (mce_chan*mask.view(-1)).mean()
    
    mse = mse_hit + mse_ener + mce_chan
    KL_div = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    graph_reg = L1 + L2
    
    # wt_reg = sum([p.abs().sum() for p in net.parameters()])
    
    return mse + KL_div + 0.1*graph_reg, mse_hit, mse_ener, mce_chan

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
        ep_loss, ep_E_mse, ep_hit_mse, ep_chan_mce, count = 0., 0., 0., 0., 0.
        for (X, NL, mask) in data_loader:
            X = X.to(device)
            mask = mask.to(device)
            A = eval_A(NL.to(device).int())*mask.unsqueeze(2)
            
            opt.zero_grad()
            loss, hit_mse, E_mse, chan_mce = loss_fn(net, X, A, mask, periodic)
            loss.backward()
            
            ep_loss += float(loss.item())
            ep_E_mse += float(E_mse.item())
            ep_hit_mse += float(hit_mse.item())
            ep_chan_mce += float(chan_mce.item())
            count += 1
            
            opt.step()
        
        if (ep+1)%10 == 0:
            torch.save(net.to("cpu").state_dict(), 
                   "Saves/Checkpoints/ep_{}.pth".format(ep+1))
        
        ep_loss /= count
        ep_hit_mse /= count
        ep_E_mse /= count
        ep_chan_mce /= count
        print("Epoch : {}".format(ep+1), 
              "Loss: {:.5f}".format(ep_loss),
              "E mse: {:.5f}".format(ep_E_mse), 
              "Hit mse: {:.5f}".format(ep_hit_mse),
              "Channel mce: {:.5f}".format(ep_chan_mce))
        
        loss_list.append(ep_loss)
        E_mse_list.append(ep_E_mse)
        hit_mse_list.append(ep_hit_mse)
    
    net.eval()
    return loss_list, E_mse_list, hit_mse_list
# %%
# net = GraphVAE()
# device = torch.device("cuda:0")
# loss2, E_mse2, hit_mse2 = train_loop(net, 1, 64, 1e-3, device, True)
# %%