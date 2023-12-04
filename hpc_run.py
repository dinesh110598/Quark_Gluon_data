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
net.load_state_dict(torch.load("Saves/ep_25.pth"))
net = net.to(device)

train_loop(net, 30, 200, 1e-3, device)
torch.save(net.to("cpu").state_dict(), "Saves/hpc_ep_30.pth")
