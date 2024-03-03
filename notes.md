## Graph VAE layers
- DenseSAGEConv: Dense version of this- 
$$x'_i = W_1 x_i + W_2.\text{mean}_{j \in N(i)} x_j$$
which basically means the weights on the normalized adjacency matrix are matrix-multiplied with the input
- The adjacency matrix are degree-normalized **every** time!
- MinCutPool
    - It uses a weight matrix $S$ which scales down the node dimension of the layer input
    - $S$ is obtained by a MLP on the layer input X, which is pretty weird for a pooling operation

## Assumptions of the model
- Is constructing A using KNN good enough? Try more neighbours?
- 
    