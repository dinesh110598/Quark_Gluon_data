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
        ecal = X[i] #[i, :, :, 1]
        xhit, yhit = torch.nonzero(ecal, as_tuple=True)
        # pos = torch.stack((xhit.float(), yhit.float()), dim=1)
        
        E = ecal[xhit, yhit]
        # Sort according to energies
        E, args = torch.sort(E, -1, True)
        xhit = xhit[args]
        yhit = yhit[args]
        
        # Node features are positions and energies of the hits
        node_ft = torch.stack((xhit.float(), 
                               yhit.float(), E), dim=1)[:400]
        # Edges are b/w k-nearest neighbors of every node
        edge_index = gnn.knn_graph(node_ft[:, :2], k=6, loop=False)
        graphs.append(torch_geometric.data.Data(
            x=node_ft, edge_index=edge_index))
        
    return graphs

# def graph_list(X: torch.Tensor) -> list:
#     """
#     Generate graph for each sample in mini-batch
#     """
#     graphs = []
    
#     for i in range(X.shape[0]):
#         tracker = X[i, :, :, 0]
#         xhit_T, yhit_T = torch.nonzero(tracker, as_tuple=True)
#         ener_T = tracker[xhit_T, yhit_T]*50
#         ft_T = torch.stack((xhit_T.float(), yhit_T.float(), ener_T), 1)
        
#         ecal = X[i, :, :, 1]
#         xhit_E, yhit_E = torch.nonzero(ecal, as_tuple=True)
#         ener_E = ecal[xhit_E, yhit_E]*50
#         ft_E = torch.stack((xhit_E.float(), yhit_E.float(), ener_E), 1)
#         # Sort according to energies
#         # E, args = torch.sort(E, -1, True)
#         # xhit = xhit[args]
#         # yhit = yhit[args]
        
#         hcal = X[i, :, :, 2]
#         xhit_H, yhit_H = torch.nonzero(hcal, as_tuple=True)
#         ener_H = hcal[xhit_H, yhit_H]*50
#         ft_H = torch.stack((xhit_H.float(), yhit_H.float(), ener_H), 1)
        
#         # Node features are positions and energies of the hits
#         node_ft = torch.cat((ft_T, ft_E, ft_H))[:1000]
        
#         # Edges are b/w k-nearest neighbors of every node
#         edge_index = gnn.knn_graph(node_ft[:, :2], k=6, loop=True)
#         layers = torch.cat((torch.full_like(ener_T, 3), 
#                            torch.full_like(ener_E, 5.5),
#                            torch.full_like(ener_H, 8.5)))[:1000]
        
#         graphs.append(torch_geometric.data.Data(
#             x=node_ft, edge_index=edge_index, layers=layers))
        
#     return graphs

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
# class Train_Dataset(torch.utils.data.Dataset):
#     def __init__(self, batch_size):
#         self.f = h5.File("quark_ecal_graph.h5")
#         self.batch = batch_size
        
#     def __len__(self):
#         return 50_000//self.batch
    
#     def __getitem__(self, i: int):
#         i1 = i*self.batch
#         i2 = (i+1)*self.batch
#         return (torch.from_numpy(self.f['X'][i1:i2]), 
#                 torch.from_numpy(self.f['A'][i1:i2]), 
#                 torch.from_numpy(self.f['mask'][i1:i2]))
    
#     def close(self):
#         self.f.close()
        
class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.f = h5.File("quark_ecal_graph.h5")
        
    def __len__(self):
        return 50_000
    
    def __getitem__(self, i: int):
        return (torch.from_numpy(self.f['X'][i]), 
                torch.from_numpy(self.f['A'][i]), 
                torch.from_numpy(self.f['mask'][i]))
    
    def close(self):
        self.f.close()
        
def get_train_dataset(L=50_000, jet="quark"):
    """
    Loads dataset to RAM. Slow to initialize.
    """
    f = h5.File(jet + "_ecal_graph_normalized.h5")
    X, NL, mask = f['X'][:L], f['NL'][:L], f['mask'][:L]
    f.close()
    return torch.utils.data.TensorDataset(torch.from_numpy(X), 
                                          torch.from_numpy(NL), 
                                          torch.from_numpy(mask))
    
def get_val_dataset(L=10_000, jet="quark"):
    f = h5.File(jet + "_val_graph_ecal.h5")
    X, NL, mask = f['X'][:L], f['NL'][:L], f['mask'][:L]
    f.close()
    return torch.utils.data.TensorDataset(torch.from_numpy(X), 
                                          torch.from_numpy(NL), 
                                          torch.from_numpy(mask))

def eval_A(NL):
    A = torch.zeros([NL.shape[0], NL.shape[1], NL.shape[1]],
                    device=NL.device)
    
    A[torch.arange(A.shape[0]).view(-1, 1, 1),
      torch.arange(A.shape[1]).view(1, -1, 1), NL] = 1/6
    return A

def encode_angle(X):
    x_hits = X[:, :, :1]
    phi = x_hits*2*torch.pi
    sin_phi = (torch.sin(phi) + 1)/2
    cos_phi = (torch.cos(phi) + 1)/2
    return torch.cat([sin_phi, cos_phi, X[:, :, 1:]], -1)

def decode_angle(X):
    sin_phi = X[:, :, :1]
    cos_phi = X[:, :, 1:2]
    x_hits = torch.atan2(sin_phi, cos_phi)/torch.pi
    return torch.cat([x_hits, X[:, :, 2:]], -1)

# def preprocess(x, device):
#     graphs = graph_list(x)
#     counts = node_counter(graphs)
#     lengs = torch.LongTensor(np.hstack(assigner(counts))).to(device)

#     compress = torch_geometric.data.Batch.from_data_list(graphs)
#     G = compress.x.clone() # All nodes of all graphs cat together
#     E = compress.edge_index.clone()
#     det = compress.layers.clone()
#     G2 = torch.cat((G, det.unsqueeze_(1)), 1)
    
#     X, mask = to_dense_batch(G2, lengs, fill_value=0, max_num_nodes=1000)
#     # X.shape = (batch, 1000, 3)
#     A = to_dense_adj(E, lengs, max_num_nodes=1000) # (batch, 1000, 1000)
    
#     return X, A, mask, counts

def preprocess(x, device):
    graphs = graph_list(x)
    counts = node_counter(graphs)
    lengs = torch.LongTensor(np.hstack(assigner(counts))).to(device)

    compress = torch_geometric.data.Batch.from_data_list(graphs)
    G = compress.x.clone() # All nodes of all graphs cat together
    E = compress.edge_index.clone()
    
    X, mask = to_dense_batch(G, lengs, fill_value=0, max_num_nodes=400)
    # X.shape = (batch, 400, 3)
    A = to_dense_adj(E, lengs, max_num_nodes=400) # (batch, 400, 400)
    
    return X, A, mask, counts
# %%