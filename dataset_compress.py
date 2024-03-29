import h5py as h5
import argparse

import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn

from graph_vae import GraphVAE
from train import train_loop
from data_load import *

min_E = 1e-3
max_E = 1e-2

with h5.File("quark-gluon_data-set_n139306.hdf5") as f:
    with h5.File("gluon-3chan-graph.h5", 'w') as g:
        X_dset = g.create_dataset('X', shape=(0, 1000, 6), dtype=np.float32,
                                  maxshape=(None, 1000, 6))
        NL_dset = g.create_dataset('NL', shape=(0, 1000, 6), dtype=np.float32,
                            maxshape=(None, 1000, 6), compression="gzip")
        mask_dset = g.create_dataset('mask', shape=(0, 1000), dtype=np.float32,
                            maxshape=(None, 1000))
        device = torch.device("cpu")
        w = 1000

        for i in range(100):
            X, y = f['X_jets'][i*w:(i+1)*w], f['y'][i*w:(i+1)*w]
            X, y = torch.from_numpy(X), torch.from_numpy(y)
            (ind_0,) = torch.nonzero(y==1, as_tuple=True)
            X = X[ind_0]
            X[X < min_E] = 0
            X[X > max_E] = 0
            
            X, A, mask, _ = preprocess(X, device)
            NL = torch.argsort(A, 1, True)[:, :6, :].int().permute(0, 2, 1)
            
            X[:, :, :2] = X[:, :, :2]/125 # Pixels in (0, 1) range
            X[:, :, 2] = X[:, :, 2]/max_E # Energies in (0, 1) range

            curr_shape = X_dset.shape[0]
            w2 = X.shape[0]
            
            X_dset.resize((curr_shape+w2, 1000, 6))
            NL_dset.resize((curr_shape+w2, 1000, 6))
            mask_dset.resize((curr_shape+w2, 1000))
            
            X_dset[curr_shape:, :, :] = X.to("cpu").numpy()
            NL_dset[curr_shape:, :, :] = NL.to("cpu").numpy()
            mask_dset[curr_shape:, :] = mask.to("cpu").numpy()