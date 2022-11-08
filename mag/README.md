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


### regcn_ns feats_type 3

```
# 45.86
python regnn_ns.py --device 6 --runs 1 --epoch 100 --residual --use_bn --train_batch_size 512 --test_batch_size 512 --lr 0.001

# 45.53
python regnn_ns.py --device 5 --runs 1 --epoch 100 --residual --use_bn --train_batch_size 512 --test_batch_size 512 --lr 0.001 --feats_type 4

# 46.93
python regnn_ns.py --device 7 --runs 1 --epoch 100 --residual --use_bn --train_batch_size 512 --test_batch_size 512 --lr 0.001 --hidden 256

# Highest Train: 63.24; Highest Valid: 48.07; Final Train: 59.95; Final Test: 47.17
python regnn_ns.py --device 7 --runs 1 --epoch 100 --residual --use_bn --train_batch_size 512 --test_batch_size 512 --lr 0.001 --hidden 512

# 30 epoch 42.98%
python regnn_ns.py --device 8 --runs 1 --epoch 100 --residual --use_bn --num_layers 3 --train_batch_size 256 --test_batch_size 256 --lr 0.001 --hidden 256

# Epoch: 53, Loss: 2.1528, Train: 43.50%, Valid: 41.59%, Test: 41.58% (收敛太慢)
python regnn_ns.py --device 7 --runs 1 --epoch 100 --residual --use_bn --train_batch_size 512 --test_batch_size 512 --lr 0.001 --hidden 512

# 
python regnn_ns.py --device 8 --runs 1 --epoch 100 --use_bn --residual --train_batch_size 1024 --test_batch_size 512 --lr 0.001 --hidden 1024

```

拿256做实验
```
# lr 0.005
# Highest Train: 54.55; Highest Valid: 47.50; Final Train: 53.68; Final Test: 46.34
python regnn_ns.py --device 7 --runs 1 --epoch 100 --residual --use_bn --train_batch_size 512 --test_batch_size 512 --lr 0.005 --hidden 256

# - bn
# Highest Train: 55.51; Highest Valid: 47.48; Final Train: 55.19; Final Test: 46.66
python regnn_ns.py --device 7 --runs 1 --epoch 100 --residual --train_batch_size 512 --test_batch_size 512 --lr 0.001 --hidden 256\

# += x_target, - residual
# Highest Train: 58.30; Highest Valid: 47.84; Final Train: 57.84; Final Test: 46.80
python regnn_ns.py --device 7 --runs 1 --epoch 100 --residual --train_batch_size 512 --test_batch_size 512 --lr 0.001 --hidden 256

# R=10 Highest Train: 55.47; Highest Valid: 46.88; Final Train: 54.72; Final Test: 46.53
python regnn_ns.py --device 9 --runs 1 --epoch 100 --use_bn --residual --train_batch_size 1024 --test_batch_size 512 --lr 0.001 --hidden 256 --scaling_factor 10
# R=1
python regnn_ns.py --device 8 --runs 1 --epoch 100 --use_bn --residual --train_batch_size 1024 --test_batch_size 512 --lr 0.001 --hidden 256 --scaling_factor 1
```

修改：REGCN+2层FC
```
# 
python regnn_ns.py --device 9 --runs 1 --epoch 100 --use_bn --residual --train_batch_size 1024 --test_batch_size 512 --lr 0.001 --hidden 256
# 1个FC
# Highest Train: 53.44 ± nan
Highest Valid: 47.94 ± nan
  Final Train: 52.96 ± nan
   Final Test: 46.98 ± nan
time used: tensor(29158.7090) tensor(nan)
python regnn_ns.py --device 7 --runs 1 --epoch 200 --use_bn --residual --train_batch_size 1024 --test_batch_size 512 --lr 0.001 --hidden 256

```

```
# 
# Highest Train: 63.52; Highest Valid: 49.91; Final Train: 61.07; Final Test: 49.08
python regnn_ns.py --device 9 --runs 1 --epoch 200 --use_bn --residual --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 4 --scaling_factor 10.

# R -> 0.000000001
# Highest Train: 61.17; Highest Valid: 49.58; Final Train: 61.09; Final Test: 48.66
python regnn_ns.py --device 8 --runs 1 --epoch 200 --use_bn --residual --train_batch_size 51
2 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 4 --scaling_factor 0.000000001

# feats_type -> 3
# Highest Train: 62.75; Highest Valid: 50.13; Final Train: 62.17; Final Test: 49.43
python regnn_ns.py --device 9 --runs 1 --epoch 200 --use_bn --residual --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10.

# feats_type -> 3; R -> 1
# Highest Train: 62.84; Highest Valid: 50.01; Final Train: 62.84; Final Test: 48.92
python regnn_ns.py --device 9 --runs 1 --epoch 200 --use_bn --residual --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10.


# feats_type -> 3; R -> 0.000000001
# Highest Train: 60.69; Highest Valid: 49.73; Final Train: 58.92; Final Test: 48.98
python regnn_ns.py --device 8 --runs 1 --epoch 200 --use_bn --residual --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 0.000000001

# feats_type -> 3; + input dropout ; (- bn)
Highest Train: 48.57; Highest Valid: 43.16; Final Train: 48.57; Final Test: 42.74


# feats_type -> 3; self_loop_type -> 2
Highest Train: 63.48; Highest Valid: 50.89; Final Train: 61.66; Final Test: 50.00
python regnn_ns.py --runs 1 --epoch 200 --residual --use_bn --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10 --self_loop_type 2 --comments ft3_sl2

# feats_type -> 3; self_loop_type -> 2; -bn
# Highest Train: 62.03; Highest Valid: 50.11; Final Train: 61.35; Final Test: 49.09
python regnn_ns.py --runs 1 --epoch 200 --residual --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10 --self_loop_type 2 --comments ft3_sl2_nobn


# feats_type -> 3; self_loop_type -> 2; addnorm
# Highest Train: 62.19; Highest Valid: 49.73; Final Train: 61.62; Final Test: 48.77
python regnn_ns.py --runs 1 --epoch 200 --residual --use_bn --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10 --self_loop_type 2 --comments ft3_sl2_addnorm


# feats_type -> 3; self_loop_type -> 2; addnormdrop(drop=0.0)
# Highest Train: 61.95; Highest Valid: 49.92; Final Train: 60.65; Final Test: 48.94
python regnn_ns.py --runs 1 --epoch 200 --residual --use_bn --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10 --self_loop_type 2 --comments ft3_sl2_addnormdrop

# feats_type -> 3; self_loop_type -> 2; somftmaxnorm
# Highest Train: 62.04; Highest Valid: 49.94; Final Train: 61.62; Final Test: 49.14
python regnn_ns.py --runs 1 --epoch 200 --residual --use_bn --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10 --self_loop_type 2 --comments ft3_sl2_softmaxnorm

# feats_type -> 3; self_loop_type -> 2; addnorm; R -> 0.000000001
# Highest Train: 60.75; Highest Valid: 48.97; Final Train: 59.34; Final Test: 48.12
python regnn_ns.py --runs 1 --epoch 200 --use_bn --residual --
train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 0.00000000010 --self_loop_type 2 --comment
s ft3_sl2_addnorm_r0
```

```
# newresidual
# Highest Train: 65.68; Highest Valid: 50.83; Final Train: 65.58; Final Test: 49.48
python regnn_ns.py --runs 1 --epoch 200 --residual --use_bn --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10 --self_loop_type 2 --comments ft3_sl2_newresidual

# 3 layer
# Highest Train: 54.52; Highest Valid: 45.69; Final Train: 54.09; Final Test: 44.73
python regnn_ns.py --runs 1 --epoch 200 --residual --use_bn --
num_layers 3 --train_batch_size 128 --test_batch_size 64 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10 --self_loop_type 2 --comm
ents ft3_sl2_layer3
```

```
# feats_type -> 3; self_loop_type -> 2
# python regnn_ns.py --runs 5 --epoch 200 --use_bn --residual --train_batch_size 512 --test_batch_size 256 --lr 0.001 --hidden 512 --feats_type 3 --scaling_factor 10 --self_loop_type 2 --comments ft3_sl2
Highest Train: 63.23 ± 0.07
Highest Valid: 50.60 ± 0.23
  Final Train: 62.30 ± 0.63
   Final Test: 49.54 ± 0.31
```

Norm4



### saint

rgcn_saint: 47.48
subgraph test

+ metapath2vec embedding: 48.54
+ 256 hidden: 49+
