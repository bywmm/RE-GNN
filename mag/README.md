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
| RGCN_SAGE     | 46.75 ± 0.57  |               |               |               | SAGE_train only   |
| REGNN_dev     | 41.22 ± 0.30  | 167.5164s     | 3             | 154,151,923   | a=1               |
| REGNN_dev     | 40.83 ± 0.55  | 162.1781s     | 3             | 154,151,923   | a=10              |
| REGNN-S       | 36.89 ± 4.70  | 164.8299s     | 3             | 154,151,923   | softmax           |
| REGNN-3       | 42.90         | 37674.5703s   | 500           |               | bn                |
| REGNN-4       | 44.69         | 37874.0664s   | 500           |               | bn                |
| REGNN-3-res   | 46.24 ± 0.35  | 14505.1260s   | 100           |               | bn-2layer         |
| REGNN-3-res   | -             |-              | 100           |               | bn-3layer-bs256   |
| REGNN-3-res   | Test-OOM      |               |               |               | bn-4layer-bs32    |
| REGNN-3-res   | 46.90 ± 0.30  | 17022.2910s   | 100           |               | bn-subgraph_test  |
| GNN-3-res     | 46.90 ± 0.38  | 16743.1699s   | 100           |               | bn-subgraph_test  |
| REGNN-3-res   | 47.28 ± 0.17  | 18156.0625s   | 100           |               | R=10              |
| REGNN-3-res   | 46.95 ± 0.34  | 18434.7578s   | 100           |               | R=100             |
| REGNN-3-res   | mag-2 gpu7    | 18434.7578s   | 100           |               | R=10  (对照)      |
| REGNN-4-res   | mag-1 gpu9    | 18434.7578s   | 100           |               | R=10             |
| REGNN-3-res   | mag gpu8      | 18434.7578s   | 100           |               | R=10 L=3 (256,256)         |
| REGNN-3-res   | mag-3 gpu5    | 18434.7578s   | 100           |               | R=10 L=4 H=256 (64,64)         |
| REGNN-SAINT   | mag-2
| REGNN-SAINT   | layer 3-4-5-6
| REGNN         | 初始化

### rgcn_ns的修改报告
- rgcn：46.64 ± 0.39
- 去掉不同的非线性，使用relation weight（R=100）：46.34 ± 0.73
- 去掉不同的非线性，使用relation weight（R=10）：46.85 ± 0.80
- 去掉不同的非线性，使用relation weight（R=1）：46.65 ± 0.67
- 去掉不同的非线性：45.97 ± 0.95

不用它的full batch test，使用NS的subgraph_test，速度会慢点
- rgcn：46.89 ± 0.62
- 去掉不同的非线性，使用relation weight（R=100）：46.90 ± 0.49
- 去掉不同的非线性，使用relation weight（R=10）：47.04 ± 0.46
- 去掉不同的非线性，使用relation weight（R=1）：46.87 ± 0.42
- 去掉不同的非线性： 46.06 ± 0.47


### regcn_ns 两种self loop实现的区别
- 加self-loop边，正常采样：
- 采样节点，加self-loop：