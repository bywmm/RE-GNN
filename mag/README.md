# ogbn-mag

This repository includes the following example scripts:

* **[MLP](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/mlp.py)**: Full-batch MLP training based on paper features and optional MetaPath2Vec features (`--use_node_embedding`). For training with MetaPath2Vec features, this script requires node embeddings be saved in `embedding.pt`. To generate them, please run `python metapath.py` [requires `torch-geometric>=1.5.0`].
* **[GNN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/gnn.py)**: Full-batch GNN training on the paper-paper relational graph using either the GCN or GraphSAGE operator (`--use_sage`) [requires `torch_geometric>=1.6.0`].
* **[R-GCN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/rgcn.py)**: Full-batch R-GCN training on the complete heterogeneous graph. This script will consume about 14GB of GPU memory [requires `torch_geometric>=1.4.3`].
* **[Cluster-GCN](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/cluster_gcn.py)**: Mini-batch R-GCN training using the Cluster-GCN algorithm [requires `torch-geometric>= 1.4.3`].
* **[NeighborSampler](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/sampler.py)**: Mini-batch R-GCN training using neighbor sampling [requires `torch-geometric>=1.5.0`].
* **[GraphSAINT](https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/graph_saint.py)**: Mini-batch R-GCN training using the GraphSAINT algorithm [requires `torch-geometric>=1.5.0`].

For the R-GCN implementation, we use distinct trainable node embeddings for all node types except for paper nodes.

## Training & Evaluation

```
# Run with default config
python graph_saint.py
```

## my results

|               | Accuracy      | Time          | Epoch         | Params        | Comments          |
| ---           | ---           | ---           | ---           | ---           | ---               |
| MLP           | 26.89 ± 0.19  | 87.2863s      | 500           | 188,509       |                   |
| GCN           | 30.43 ± 0.18  | 44.99s        | 100           | 122,717       |                   |
| SAGE          | 31.29 ± 0.19  | 39.60s        | 100           | 244,829       |                   |
| GCN-SAGE      | 36.56 ± 0.31  | 155.9268s     | 3             | 154,090,845   |                   |
| GCN-SAGE      | 36.62 ± 0.41  | 1034.2740s    | 20            | 154,090,845   |                   |
| RGCN          | OOM           |               |               | 154,366,772   |                   |
| RGCN-Saint    | 47.73 ± 0.36  | 337.103s      | 30            | 154,366,772   | SAINT-train only  |
| RGCN_SAGE     | 45.11 ± 0.44  | 214.7822s     | 3             | 154,366,772   | SAGE_train & test |
| REGNN_dev     | 41.22 ± 0.30  | 167.5164s     | 3             | 154,151,923   | a=1               |
| REGNN_dev     | 40.83 ± 0.55  | 162.1781s     | 3             | 154,151,923   | a=10              |
| REGNN-S       | 36.89 ± 4.70  | 164.8299s     | 3             | 154,151,923   | softmax           |
| REGNN-3       | 42.90         | 37674.5703s   | 500           |               | bn                |
| REGNN-4       | 44.69         | 37874.0664s   | 500           |               | bn                |
| REGNN-3-res   | 45.75 ± 0.36  | 8105.6509s    | 100           |               | bn-2layer         |
| REGNN-3-res   | -             |-              | 100           |               | bn-3layer-bs256   |
| REGNN-3-res   | Test-OOM      |               |               |               | bn-4layer-bs32    |
