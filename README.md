# Optimization and Interpretability of Graph Attention Networks for Small Sparse Graph Structures in Automotive Applications

This repository is the official implementation of [Optimization and Interpretability of Graph Attention Networks for Small Sparse Graph Structures in Automotive Applications](https://ieeexplore.ieee.org/document/10186536) (https://arxiv.org/abs/2305.16196). 
The implementation of GAT+ is based on the PyTorch Geometric library, enabling users to effortlessly integrate and apply the proposed architecture to a wide range of applications. It is created using the PyTorch Geometric MessagePassing base class [torch_gemetric.nn.conv.MessagePassing](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing).

GAT+, a variant of Graph Attention Networks (GATs), enhances training robustness and performance by optimizing gradient flow. In GAT+ the attention scores are computed using the Sofplus activation function instead of LeakyRELU:
```math
\alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{Softplus}\left(\mathbf{\Theta}_R\,
        \mathbf{x}_i + \mathbf{\Theta}_L \,\mathbf{x}_j
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) }
        \exp\left(\mathbf{a}^{\top}\mathrm{Softplus}\left(\mathbf{\Theta}_R\,
        \mathbf{x}_i + \mathbf{\Theta}_L \,\mathbf{x}_k
        \right)\right)}.
```
If the graph has multi-dimensional edge features, the attention coefficients $\alpha_{i,j}$ are computed as
```math
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{Softplus}\left(\mathbf{\Theta}_R\,
        \mathbf{x}_i + \mathbf{\Theta}_L \,\mathbf{x}_j + \mathbf{\Theta}_E \,\mathbf{e}_{i,j}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) }
        \exp\left(\mathbf{a}^{\top}\mathrm{Softplus}\left(\mathbf{\Theta}_R\,
        \mathbf{x}_i + \mathbf{\Theta}_L\, \mathbf{x}_k + \mathbf{\Theta}_E \, \mathbf{e}_{i,k}
        \right)\right)}.
```
Depending on the defined operation mode, the features are updated as follows:  <br>
- **'att'** (default)
```math
 \mathbf{x}_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{\Theta}_L \,\mathbf{x}_j 
```
- **'theta_n'** 
```math
 \mathbf{x}_i' = \mathbf{\Theta}_n \,\mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{\Theta}_L \,\mathbf{x}_j \quad \quad (i \notin \mathcal{N}(i))
```
- **'theta_r'** 
```math
 \mathbf{x}_i' = \mathbf{\Theta}_R \,\mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{\Theta}_L \,\mathbf{x}_j \quad \quad (i \notin \mathcal{N}(i))
```
## Parameters
- **in_channels** (int or tuple) – Size of each input sample, or -1 to derive the size from the first input(s) to the forward method. A tuple corresponds to the sizes of source and target dimensionalities.
- **out_channels** (int) – Size of each output sample.
- **heads** (int, optional) – Number of multi-head-attentions. (default: 1)
- **concat** (bool, optional) – If set to False, the multi-head attentions are averaged instead of concatenated. (default: True)
- **mode** (str, optional) – Set operation mode of update function. Set to either 'att', 'theta_n' or 'theta_r'. If mode is set to 'theta_n', the query node features are weighted by and addtional linear layer in the update function. (default: 'att')
- **dropout** (float, optional) – Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training. (default: 0)
- **self_loop** (bool, optional): Self loop is considered in the node update. If 'False', the query node information is not considered. (default: True)
- **edge_dim** (int, optional) – Edge feature dimensionality (in case there are any). (default: None)
- **fill_value** (float or torch.Tensor or str, optional) – The way to generate edge features of self-loops (in case edge_dim != None). If given as float or torch.Tensor, edge features of self-loops will be directly given by fill_value. If given as str, edge features of self-loops are computed by aggregating all features of edges that point to the specific node, according to a reduce operation. ("add", "mean", "min", "max", "mul"). (default: "mean")
- **bias** (bool, optional) – If set to False, the layer will not learn an additive bias. (default: True)
- **share_weights** (bool, optional) – If set to True, the same matrix will be applied to the source and the target node of every edge. (default: False)
- ****kwargs** (optional) – Additional arguments of conv.MessagePassing.

## Shapes
- **input**: node features $(\vert \mathcal{V} \vert, F_{in})$ or $\left[(\vert \mathcal{V}_s \vert, F_s), (\vert \mathcal{V}_t \vert, F_t)\right]$ if bipartite, edge indices $(2, \vert\mathcal{E}\vert)$, edge features $(\vert\mathcal{E}\vert, D)$ (optional).
- **output**: node features $(\vert \mathcal{V} \vert, H * F_{out})$ or  if bipartite. If return_attention_weights=True, then  or  if bipartite

## Run
```
forward (x: Union[Tensor, Tuple[Tensor, Tensor], edge_index: Union[Tensor, SparseTensor],
         edge_attr: Optional[Tensor] = None, return_attention_weights: Optional[bool] = None)
```
Runs the forward pass of the module. 
- **return_attention_weights** (bool, optional) – If set to True, will additionally return the tuple (edge_index, attention_weights), holding the computed attention weights for each edge. (default: None)

## Citation
[Optimization and Interpretability of Graph Attention Networks for Small Sparse Graph Structures in Automotive Applications](https://ieeexplore.ieee.org/document/10186536)
```
@inproceedings{gatpconv23,
  author={Neumeier, Marion and Tollkühn, Andreas and Dorn, Sebastian and Botsch, Michael and Utschick, Wolfgang},
  booktitle={2023 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={Optimization and Interpretability of Graph Attention Networks for Small Sparse Graph Structures in Automotive Applications}, 
  year={2023},
  doi={10.1109/IV55152.2023.10186536}}
```
