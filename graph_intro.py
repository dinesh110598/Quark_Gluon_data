# %%
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
# %%
op = gnn.DenseSAGEConv(1, 1, bias=False)
print([(name, param) for name, param in op.named_parameters()])
# %%
with torch.no_grad():
    op.lin_rel.weight = nn.Parameter(torch.ones([1, 1]))
    op.lin_root.weight = nn.Parameter(torch.ones([1, 1]))
# %%
X = (torch.arange(1, 5)**2).unsqueeze(1).float()
A = torch.FloatTensor(
    [[0, 1, 1, 0],
     [1, 0, 1, 0],
     [1, 1, 0, 0],
     [1, 0, 1, 0]
    ])
# %%
X2 = op(X, A)
# %%