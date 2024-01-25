import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

from graph_vae import GraphVAE
from train import train_loop, loss_infer
from data_load import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

net = GraphVAE()
net = net.to(device)

loss, E_mse, hit_mse = train_loop(net, 60, 256, 1e-3, device, False)
torch.save(net.to("cpu").state_dict(), "Saves/hpc_ep_60.pth")
