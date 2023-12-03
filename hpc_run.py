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

net = GraphVAE()
net.load_state_dict(torch.load("Saves/L_50k_ep_15.pth"))
train_loop(net, 30, 200)

torch.save(net.state_dict(), "Saves/L_50k_ep_45.pth")
