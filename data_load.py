# %%
import h5py as h5
import numpy as np
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric import nn as gnn
from torch_geometric.utils import to_dense_batch, to_dense_adj
# %%
def graph_list(X: torch.Tensor) -> list:
    """
    Generate graph for each sample in mini-batch
    """
    graphs = []
    
    for i in range(X.shape[0]):
        ecal = X[i, :, :, 1]
        xhit, yhit = torch.nonzero(ecal, as_tuple=True)
        # pos = torch.stack((xhit.float(), yhit.float()), dim=1)
        
        E = ecal[xhit, yhit]*50
        # Sort according to energies
        # E, args = torch.sort(E, -1, True)
        # xhit = xhit[args]
        # yhit = yhit[args]
        
        # Node features are positions and energies of the hits
        node_ft = torch.stack((xhit.float(), 
                               yhit.float(), E), dim=1)[:300]
        # Edges are b/w k-nearest neighbors of every node
        edge_index = gnn.knn_graph(node_ft[:, :2], k=6, loop=True)
        graphs.append(torch_geometric.data.Data(
            x=node_ft, edge_index=edge_index))
        
    return graphs

## generate list to count nodes for each graph
def node_counter(samples):
    inds=[]
    for k in samples:
        inds.append(k['x'].shape[0])
    return inds

def assigner(counts):
  fin=[]
  countit=0
  for m in counts:
      fin.append(np.repeat(countit,m))
      countit+=1
  return fin
# %%        
class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, batch_size):
        self.f = h5.File("quark-gluon_data-set_n139306.hdf5")
        self.batch = batch_size
        
    def __len__(self):
        return len(self.f['y'])//self.batch
    
    def __getitem__(self, i: int):
        i1 = i*self.batch
        i2 = (i+1)*self.batch
        return (torch.from_numpy(self.f['X_jets'][i1:i2]), 
                torch.from_numpy(self.f['m0'][i1:i2]), 
                torch.from_numpy(self.f['pt'][i1:i2]), 
                torch.from_numpy(self.f['y'][i1:i2]))
    
    def close(self):
        self.f.close()
        
def get_train_dataset(L=100_000):
    """
    Loads dataset to RAM. Slow to initialize.
    """
    f = h5.File("quark-gluon_data-set_n139306.hdf5")
    x, y = f['X_jets'][:L], f['y'][:L]
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    (ind_0,) = torch.nonzero(y==0, as_tuple=True)
    x = x[ind_0]
    f.close()
    return torch.utils.data.TensorDataset(x)

def preprocess(x, device):
    graphs = graph_list(x)
    counts = node_counter(graphs)
    lengs = torch.LongTensor(np.hstack(assigner(counts))).to(device)

    compress = torch_geometric.data.Batch.from_data_list(graphs)
    G = compress.x.clone() # All nodes of all graphs cat together
    E = compress.edge_index.clone()
    
    X, mask = to_dense_batch(G, lengs, fill_value=0, max_num_nodes=300)
    # X.shape = (batch, 300, 3)
    A = to_dense_adj(E, lengs, max_num_nodes=300) # (batch, 300, 300)
    
    return X, A, mask, counts

def reconstruct_img(Y, counts):
    xhit, yhit = Y[:, :, 0], Y[:, :, 1]
    val = Y[:, :, 2]/50.
    
    xhit = (xhit % 125).int()
    yhit = (yhit % 125).int()
    
    ecal = torch.zeros((Y.shape[0], 125, 125))
    for j in range(Y.shape[0]):
        # Add fancy/optimized indexing later
        for i in range(counts[j]):
            ecal[j, xhit[j, i], yhit[j, i]] += 0.1# val[j, i]
    
    return ecal
# %%