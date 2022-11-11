import argparse
from tqdm import tqdm
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ModuleDict, Parameter, init, ParameterDict
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
from utils import weighted_degree, get_self_loop_index, softmax
from utils import MsgNorm, args_print
from early_stopping import EarlyStopping
from regnn_layers import REGCNConv, REGATConv, REGATv2Conv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print('Using CUDA')
        torch.cuda.manual_seed(seed)

set_seed(123)

# torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='OGBN-MAG (REGCN-NS)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model', type=str, default='regcn', help='regcn, regat, regatv2')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--early_stop', type=int, default=50)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=1024)
parser.add_argument('-r', '--scaling_factor', type=float, default=10.)
parser.add_argument('--feats_type', type=int, default=3,
                    help='Type of the node features used. ' + 
                         '1 - target node features (zero vec for others); ' +
                         '2 - target node features (id vec for others); ' +
                         '3 - target node features (random features for others); ' +
                         '4 - target node features (Complex emb for others);' + 
                         '? - all id vec.'
                    ) # Note that OGBN-MAG only has target node features.
parser.add_argument('--residual', action='store_true', default=False)
parser.add_argument('--gcn', action='store_true', default=False)
parser.add_argument('--use_norm', type=str, default='ln', help='non, bn, or ln')
parser.add_argument('--self_loop_type', type=int, default=1,
                    help='1 - add self-loop connections and then sample it as edges;' + 
                         '2 - sample nodes and then add self-loop')
parser.add_argument('--use_scheduler', action='store_true')
parser.add_argument('--comments', type=str, default='raw')

args = parser.parse_args()
assert args.use_norm in ['non', 'bn', 'ln']
args_print(args)

init_st = time.time()

home_dir = os.getenv("HOME")
root = os.path.join(home_dir, "dataset/graph/OGB")
dataset = PygNodePropPredDataset(root=root, name='ogbn-mag')
data = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
logger = Logger(args.runs, args)

# We do not consider those attributes for now.
data.node_year_dict = None
data.edge_reltype_dict = None

print(data)

edge_index_dict = data.edge_index_dict

# We need to add reverse edges to the heterogeneous graph.
r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('author', 'writes', 'paper')]
edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

# Convert to undirected paper <-> paper relation.
edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
edge_index_dict[('paper', 'cites', 'paper')] = edge_index

if args.self_loop_type == 1:
    # Add Self-Loop Relation
    self_loop_index = get_self_loop_index(num_node=data.num_nodes_dict['author'])
    edge_index_dict[('author', 'selfloop', 'author')] = self_loop_index

    self_loop_index = get_self_loop_index(num_node=data.num_nodes_dict['field_of_study'])
    edge_index_dict[('field_of_study', 'selfloop', 'field_of_study')] = self_loop_index

    self_loop_index = get_self_loop_index(num_node=data.num_nodes_dict['institution'])
    edge_index_dict[('institution', 'selfloop', 'institution')] = self_loop_index

    self_loop_index = get_self_loop_index(num_node=data.num_nodes_dict['paper'])
    edge_index_dict[('paper', 'selfloop', 'paper')] = self_loop_index


# del edge_index_dict[('author', 'affiliated_with', 'institution')]
# del edge_index_dict[('institution', 'to', 'author')]
# del edge_index_dict[('author', 'writes', 'paper')]
# del edge_index_dict[('paper', 'to', 'author')]
# del edge_index_dict[('author', 'selfloop', 'author')]
# del edge_index_dict[('institution', 'selfloop', 'institution')]
# print(edge_index_dict.keys())


# We convert the individual graphs into a single big one, so that sampling
# neighbors does not need to care about different edge types.
# This will return the following:
# * `edge_index`: The new global edge connectivity.
# * `edge_type`: The edge type for each edge.
# * `node_type`: The node type for each node.
# * `local_node_idx`: The original index for each node.
# * `local2global`: A dictionary mapping original (local) node indices of
#    type `key` to global ones.
# `key2int`: A dictionary that maps original keys to their new canonical type.
out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

# Map informations to their canonical type.
# only target nodes (Paper) have raw features
x_dict = {}
target_node_type = None
for key, x in data.x_dict.items():
    x_dict[key2int[key]] = x
    target_node_type = key2int[key]

num_nodes = 0
num_nodes_dict = {}
num_feature_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N
    num_nodes += N
        # 1 - target node features (zero vec for others)
    if args.feats_type == 1:
        if key2int[key] != target_node_type:
            x_dict[key2int[key]] = torch.zeros(N, 128)
        num_feature_dict[key2int[key]] = 128
    # 2 - target node features (id vec for others)
    elif args.feats_type == 2:
        # Using Node Embeding
        # indices = np.vstack((np.arange(N), np.arange(N)))
        # indices = torch.LongTensor(indices)
        # values = torch.FloatTensor(np.ones(N))
        # x_dict[key2int[key]] = torch.sparse.FloatTensor(indices, values, torch.Size([N, N])).to_dense()
        num_feature_dict[key2int[key]] = 128
    # 3 - target node features (random features for others)
    elif args.feats_type == 3:
        if key2int[key] != target_node_type:
            x_dict[key2int[key]] = torch.Tensor(N, 128).uniform_(-0.5, 0.5)
        num_feature_dict[key2int[key]] = 128
    # 4 - Use extra embeddings generated with the Complex method
    elif args.feats_type == 4:
        if key2int[key] != target_node_type:
            home_dir = os.getenv("HOME")
            path = os.path.join(home_dir, "projects/gcns/15-heterogeneous/SeHGNN/data/complex_nars")
            x_dict[key2int[key]] = torch.load(os.path.join(path, key+'.pt'), map_location=torch.device('cpu')).float()
            num_feature_dict[key2int[key]] = x_dict[key2int[key]].size(1)
        else:
            num_feature_dict[key2int[key]] = 128
if args.feats_type == 5:
    nodes_embedding_path = "data/mag_embedding.pt"
    embedding_dict = torch.load(nodes_embedding_path, map_location='cpu')
    print(f"{nodes_embedding_path} is loaded successfully.")
    for key, N in data.num_nodes_dict.items():
        if key2int[key] != target_node_type:
            x_dict[key2int[key]] = embedding_dict[key]
        else:
            x_dict[key2int[key]] = torch.cat([x_dict[key2int[key]], embedding_dict[key]], dim=1)
        num_feature_dict[key2int[key]] = x_dict[key2int[key]].size(1)
           
# paper training nodes.
paper_idx = local2global['paper']
paper_train_idx = paper_idx[split_idx['train']['paper']]

if args.num_layers == 2:
    tr_size = [25, 20]           # batch_size 1024
elif args.num_layers == 3:
    tr_size = [20, 15, 10]       # batch_size 256
elif args.num_layers == 4:
    tr_size = [20, 15, 10, 10]   # batch_size 32
train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=tr_size, batch_size=args.train_batch_size, shuffle=True,
                               num_workers=4)
test_loader = NeighborSampler(edge_index, node_idx=paper_idx,
                               sizes=tr_size, batch_size=args.test_batch_size, shuffle=False,
                               num_workers=4)
subgraph_loader = NeighborSampler(edge_index, node_idx=None, 
                               sizes=[-1], batch_size=args.test_batch_size, shuffle=False,
                               num_workers=4)

class REGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_layers, scaling_factor,
                 dropout, num_feature_dict, num_edge_types, residual, gcn, use_norm=None):
        super(REGNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.use_norm = use_norm

        node_types = list(num_feature_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        if args.model == 'regcn':
            self.hidden_dim = hidden_channels
        else:
            self.hidden_dim = hidden_channels * heads
        if args.feats_type == 2:
            self.emb_dict = ParameterDict({
                f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
                for key in set(node_types).difference(set([target_node_type]))
            })
            self.lin = Linear(in_channels, self.hidden_dim)
        else:
            # Feature Projection
            self.lins = ModuleDict()
            for key in set(node_types): 
                self.lins[str(key)] = Linear(num_feature_dict[key], self.hidden_dim)

        # REGNNConv
        self.convs = ModuleList()
        if args.model == 'regcn':
            for _ in range(num_layers):
                self.convs.append(
                    REGCNConv(hidden_channels, hidden_channels, num_node_types, num_edge_types, scaling_factor, gcn, 
                        dropout=dropout, residual=residual, use_norm=self.use_norm, self_loop_type=args.self_loop_type)
                )
        elif args.model == 'regat':
            for _ in range(num_layers):
                self.convs.append(
                    REGATConv(self.hidden_dim, hidden_channels, num_node_types, num_edge_types, heads, scaling_factor , 
                        dropout=dropout, residual=residual, use_norm=self.use_norm, self_loop_type=args.self_loop_type)
                )
        elif args.model == 'regatv2':
            for _ in range(num_layers):
                self.convs.append(
                    REGATv2Conv(self.hidden_dim, hidden_channels, num_node_types, num_edge_types, heads, scaling_factor , 
                        dropout=dropout, residual=residual, use_norm=self.use_norm, self_loop_type=args.self_loop_type)
                )
        else:
            raise NotImplementedError
        
        # self.convs.append(REGCNConv(hidden_channels, out_channels, num_node_types, num_edge_types, scaling_factor, gcn))
        self.out_lin = Linear(self.hidden_dim, out_channels)
        if self.use_norm  == 'bn':
            self.norm = torch.nn.BatchNorm1d(self.hidden_dim)
        elif self.use_norm == 'ln':
            self.norm = torch.nn.LayerNorm(self.hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        if args.feats_type == 2:
            for emb in self.emb_dict.values():
                torch.nn.init.xavier_uniform_(emb)
            self.lin.reset_parameters()
        else:
            for lin in self.lins.values():
                lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()
        self.out_lin.reset_parameters()
        if self.use_norm in ['bn', 'ln']:
            self.norm.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None, device=None):
        # Create global node feature matrix.
        if n_id is not None:
            node_type = node_type[n_id]
            local_node_idx = local_node_idx[n_id]
        
        if args.feats_type == 2:
            t = torch.zeros((node_type.size(0), self.in_channels),
                            device=device)
            for key, x in x_dict.items():
                mask = node_type == key
                t[mask] = x[local_node_idx[mask]].to(device)
            for key, emb in self.emb_dict.items():
                mask = node_type == int(key)
                t[mask] = emb[local_node_idx[mask]].to(device)
            h = self.lin(t.to(node_type.device)).to(device)
        else:
            h = torch.zeros((node_type.size(0), self.hidden_dim),
                            device=device)
            for key, x in x_dict.items():
                mask = node_type == key
                # It will make the training or inference speed slower.
                x_tmp = x[local_node_idx[mask]].to(node_type.device)
                h_tmp = self.lins[str(key)](x_tmp)
                h[mask] = h_tmp.to(device)

        return h

    def forward(self, n_id, x_dict, adjs, edge_type, node_type, local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx, n_id, node_type.device)
        node_type = node_type[n_id]
        # if self.use_norm in ['bn', 'ln']:
        #     x = self.norm(x)

        # x = F.dropout(x, p=self.dropout, training=self.training)

        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target node embeddings.
            node_type = node_type[:size[1]]  # Target node types.
            conv = self.convs[i]
            x = conv((x, x_target), edge_index, edge_type[e_id], node_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_lin(x)

        return x.log_softmax(dim=-1)
        
    def inference(self, x_dict, subgraph_loader, edge_type, node_type, local_node_idx, device):
        x_all = self.group_input(x_dict, node_type, local_node_idx, device=torch.device('cpu'))
        # if self.use_norm in ['bn', 'ln']:
        #     x_all = self.norm(x_all.to(device)).to('cpu')

        for layer in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                node_type_src = node_type[n_id]
                node_type_target = node_type_src[:size[1]]
                x = self.convs[layer]((x, x_target), edge_index, edge_type[e_id], node_type_target)
                x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)
        x = x_all.to(device)
        x = self.out_lin(x)

        return x


device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = REGNN(128, args.hidden_channels, dataset.num_classes, args.heads, args.num_layers, args.scaling_factor,
             args.dropout, num_feature_dict, len(edge_index_dict.keys()), args.residual, args.gcn, args.use_norm).to(device)

sum_p = sum(p.numel() for p in model.parameters())
print("Num params:", sum_p)

# Create global label vector.
y_global = node_type.new_full((node_type.size(0), 1), -1)
y_global[local2global['paper']] = data.y_dict['paper']

# Move everything to the GPU.
# x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_type = edge_type.to(device)
node_type = node_type.to(device)
local_node_idx = local_node_idx.to(device)
y_global = y_global.to(device)


def train(epoch, optimizer, scheduler, train_steps):
    model.train()

    # pbar = tqdm(total=paper_train_idx.size(0))
    # pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y = y_global[n_id][:batch_size].squeeze()
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        train_steps += 1
        if scheduler is not None:
            # scheduler.step(train_steps)
            scheduler.step()
        # pbar.update(batch_size)

    # pbar.close()

    loss = total_loss / paper_train_idx.size(0)

    return loss

@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x_dict, subgraph_loader, edge_type, node_type, local_node_idx, device=device)
    y_pred = out.argmax(-1, keepdim=True)[local2global['paper']]
    y_true = y_global[local2global['paper']]

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc

@torch.no_grad()
def test_tmp():
    model.eval()

    pbar = tqdm(total=paper_idx.size(0))
    pbar.set_description(f'Test')

    y_pred, y_true = [], []
    for batch_size, n_id, adjs in test_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y_t = y_global[n_id][:batch_size]
        y_p = out.argmax(dim=-1, keepdim=True).cpu()
        y_pred.append(y_p)
        y_true.append(y_t)
        pbar.update(batch_size)

    pbar.close()
    
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc

print(f"Init time: {time.time()-init_st:.2f}s")
time_used = []
# test()  # Test if inference on GPU succeeds.
for run in range(args.runs):

    save_model_folder = f'checkpoint/REGNN_NS'
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)
    save_model_path = os.path.join(save_model_folder, f"REGNN_NS-{args.comments}-{run + 1:02d}.pkl")

    st = time.perf_counter()
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch=len(train_loader)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * args.epochs, eta_min=args.lr / 100)
    else:
        scheduler = None
    train_steps = 0
    best_valid_acc = -1.0
    epoch_times = []
    es = EarlyStopping(args.early_stop)
    for epoch in range(1, 1 + args.epochs):
        epoch_st = time.time()
        loss = train(epoch, optimizer, scheduler, train_steps)
        result = test()
        logger.add_result(run, result)
        train_acc, valid_acc, test_acc = result
        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), save_model_path)
        es(valid_acc)
        if es.early_stop:
            break
        epoch_time = time.time() - epoch_st
        epoch_times.append(epoch_time)
        print(f'Run: {run + 1:02d}, '
              f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}%, '
              f'Test: {100 * test_acc:.2f}%, '
              f'Epoch time: {epoch_time:.2f}s')
        
    epoch_times = np.array(epoch_times)
    print(f'Average Epoch Time: {epoch_times.mean():.2f}s ± {epoch_times.std():.2f}')
    time_used.append(time.perf_counter()-st)
    logger.print_statistics(run)
logger.print_statistics()
time_used = torch.tensor(time_used)
print("time used:", time_used.mean(), time_used.std())