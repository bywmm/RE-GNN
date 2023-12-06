
# RE-GNN
Code for Paper "Enabling Homogeneous GNNs to Handle Heterogeneous Graphs via Relation Embedding", IEEE TBD, 2023.

## Reproduce the results.

### Step 1: Install requirements.

```txt
torch==1.7.0
networkx==2.4
tqdm==4.46.0
numpy==1.16.2
scipy==1.4.1
dgl==0.7.1
scikit_learn==1.0.2
```


### Step 2: Get the reported results.

The scripts of our main results are provided in the `scripts` file.
Run one of them to get the reported results.
For example, 

```bash
bash scripts/dblp_regcn_res.sh $GPU_ID$
```


### Citation

If you find our codes useful or get inspirations from our research, please consider citing our work.

```
@article{regnn,
  title={Enabling Homogeneous GNNs to Handle Heterogeneous Graphs Via Relation Embedding},
  author={Wang, Junfu and Guo, Yuanfang and Yang, Liang and Wang, Yunhong},
  journal={IEEE Transactions on Big Data},
  year={2023},
  volume={9},
  pages={1697--1710},
}
```