
# RE-GNN
This is the anonymous repository for paper submission 540 (Relation Embedding based Graph Neural Networks) for KDD-22.

## Using RE-GNN respository to reproduce the results.

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


### Step 2: Run scripts

The scripts of our main results are provided.
Run one of them to get the reported results.
For example, 

```bash
bash script_imdb_regcn.sh $GPU_ID$
```


Note that since the sizes of ACM and DBLP datasets are large, we do not add them to this repository.
It can be founded on previous work, which is mentioned in the paper submission.

<!-- 
|       |DBLP   |       |ACM    |       |IMDB   |       |
|---    |---    |---    |---    |---    |---    |---    |
|RE-GAT |94.43  |94.83  |93.65  |93.57  |61.04  |61.34  |
|RE-GCN |95.52  |95.81  |94.54  |94.47  |61.54  |61.83  | -->