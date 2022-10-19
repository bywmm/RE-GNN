from copy import copy
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ModuleDict, Parameter, init
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborSampler, GraphSAINTRandomWalkSampler
from torch_geometric.data import Data
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing, GCNConv
from utils import weighted_degree, get_self_loop_index, softmax
import numpy as np
import os

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

parser = argparse.ArgumentParser(description='OGBN-MAG')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=10000)
parser.add_argument('--test_batch_size', type=int, default=1024)
parser.add_argument('--walk_length', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=30)
parser.add_argument('--scaling_factor', type=float, default=1.)
parser.add_argument('--test_step', type=int, default=1)
parser.add_argument('--feats_type', type=int, default=3,
                    help='Type of the node features used. ' + 
                         '1 - target node features (zero vec for others); ' +
                         '2 - target node features (id vec for others); ' +
                         '3 - target node features (random features for others); ' +
                         '4 - target node features (Complex emb for others);' + 
                         '? - all id vec.'
                    ) # Note that OGBN-MAG only has target node features.
parser.add_argument('--use_bn', action='store_true', default=False)
parser.add_argument('--residual', action='store_true', default=False)
parser.add_argument('--gcn', action='store_true', default=False)

args = parser.parse_args()
print(args)

root = '/home/wangjunfu/dataset/graph/OGB'
dataset = PygNodePropPredDataset(root=root, name='ogbn-mag')
data = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
logger = Logger(args.runs, args)

# print(data)

# We do not consider those attributes for now.
data.node_year_dict = None
data.edge_reltype_dict = None

# print(data)

edge_index_dict = data.edge_index_dict

# Add reverse edges to the heterogeneous graph.
r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('author', 'writes', 'paper')]
edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

# Convert to undirected paper <-> paper relation.
edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
edge_index_dict[('paper', 'cites', 'paper')] = edge_index

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
    target_node_type = key
    x_dict[key2int[key]] = x

num_nodes = 0
num_nodes_dict = {}
num_feature_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N
    num_nodes += N
    if key != target_node_type:
        # 1 - target node features (zero vec for others)
        if args.feats_type == 1:
            x_dict[key2int[key]] = torch.zeros(N, 128)
            num_feature_dict[key2int[key]] = 128
        # 2 - target node features (id vec for others)
        elif args.feats_type == 2:
            indices = np.vstack((np.arange(N), np.arange(N)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(N))
            x_dict[key2int[key]] = torch.sparse.FloatTensor(indices, values, torch.Size([N, N]))
            num_feature_dict[key2int[key]] = N
            
        # 3 - target node features (random features for others)
        elif args.feats_type == 3:
            x_dict[key2int[key]] = torch.Tensor(N, 128).uniform_(-0.5, 0.5)
            num_feature_dict[key2int[key]] = 128
        # 4 - Use extra embeddings generated with the Complex method
        elif args.feats_type == 4:
            home_dir = os.getenv("HOME")
            path = os.path.join(home_dir, "projects/gcns/15-heterogeneous/SeHGNN/data/complex_nars")
            x_dict[key2int[key]] = torch.load(os.path.join(path, key+'.pt'), map_location=torch.device('cpu')).float()
            num_feature_dict[key2int[key]] = x_dict[key2int[key]].size(1)
    else:
        num_feature_dict[key2int[key]] = 128

homo_data = Data(edge_index=edge_index, edge_attr=edge_type,
                 node_type=node_type, local_node_idx=local_node_idx,
                 num_nodes=node_type.size(0))

# paper training nodes.
paper_idx = local2global['paper']
paper_train_idx = paper_idx[split_idx['train']['paper']]

homo_data.y = node_type.new_full((node_type.size(0), 1), -1)
homo_data.y[paper_idx] = data.y_dict['paper']

homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
homo_data.train_mask[paper_train_idx] = True

if args.walk_length == -1:
    args.walk_length = args.num_layers
train_loader = GraphSAINTRandomWalkSampler(homo_data,
                                           batch_size=args.train_batch_size,
                                           walk_length=args.walk_length,
                                           num_steps=args.num_steps,
                                           sample_coverage=0,
                                           save_dir=dataset.processed_dir)
subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                               sizes=[-1], batch_size=args.test_batch_size, shuffle=False,
                               num_workers=12)

class REGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types, scaling_factor=100., gcn=False, dropout=0., 
                 use_softmax=False):
        super(REGCNConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_softmax = use_softmax
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_root = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        if gcn:
            self.relation_weight = Parameter(torch.Tensor(num_edge_types), requires_grad=False)
        else:
            self.relation_weight = Parameter(torch.Tensor(num_edge_types), requires_grad=True)
        self.scaling_factor = scaling_factor

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)
        init.constant_(self.relation_weight, 1.0 / self.scaling_factor)

    def forward(self, x, edge_index, edge_type, return_weights=False):
        # shape of x: [N, in_channels]
        # shape of edge_index: [2, E]
        # shape of edge_type: [E]
        # shape of e_feat: [E, edge_tpes+node_types]

        edge_type = edge_type.view(-1, 1)
        e_feat = torch.zeros(edge_type.shape[0], self.num_edge_types, device=edge_type.device).scatter_(1, edge_type.view(-1, 1), 1.0)
        
        if isinstance(x, torch.Tensor):
            x = (x, x)
        x_src, x_target = x
        x_src = torch.matmul(x_src, self.weight)
        x_target = torch.matmul(x_target, self.weight_root)
        x = (x_src, x_target)

        # Cal edge weight according to its relation type
        relation_weight = self.relation_weight * self.scaling_factor
        relation_weight = F.leaky_relu(relation_weight)
        # print(relation_weight)
        edge_weight = torch.matmul(e_feat, relation_weight)  # [E]

        # Compute GCN normalization
        row, col = edge_index

        # self.use_softmax = True
        if self.use_softmax:
            ew = softmax(edge_weight, col)
        else:
            # mean aggregator
            deg = weighted_degree(col, edge_weight, x_target.size(0), dtype=x_target.dtype).abs()
            deg_inv = deg.pow(-1.0)
            norm = deg_inv[col]
            ew = edge_weight * norm
        
        ew = F.dropout(ew, p=self.dropout, training=self.training)
        out = self.propagate(edge_index, x=x, ew=ew)

        if return_weights:
            return out, ew
        else:
            return out

    def message(self, x_j, ew):

        return ew.view(-1, 1) * x_j

    def update(self, aggr_out):

        aggr_out = aggr_out + self.bias

        return aggr_out


class REGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, scaling_factor,
                 dropout, num_feature_dict, num_edge_types, use_bn, residual, gcn):
        super(REGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_bn = use_bn
        self.residual = residual

        node_types = list(num_feature_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Feature Projection
        self.lins = ModuleDict()
        for key in set(node_types): 
            self.lins[str(key)] = Linear(num_feature_dict[key], hidden_channels)

        # REGNNConv
        self.convs = ModuleList()
        self.convs.append(REGCNConv(hidden_channels, hidden_channels, num_node_types, num_edge_types, scaling_factor, gcn))
        for _ in range(num_layers - 2):
            self.convs.append(REGCNConv(hidden_channels, hidden_channels, num_node_types, num_edge_types, scaling_factor, gcn))
        self.convs.append(REGCNConv(hidden_channels, out_channels, self.num_node_types, num_edge_types, scaling_factor, gcn))

        self.prelus = ModuleList()
        for _ in range(num_layers-1):
            self.prelus.append(torch.nn.PReLU())

        self.bns = ModuleList()
        for _ in range(num_layers-1):
            self.bns.append(torch.nn.BatchNorm1d(self.hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins.values():
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None, device=None):
        # Create global node feature matrix.
        # the device of node_type is 
        if n_id is not None:
            node_type = node_type[n_id]
            local_node_idx = local_node_idx[n_id]
        
        if device is None:
            device = node_type.device

        h = torch.zeros((node_type.size(0), self.hidden_channels),
                        device=device)

        for key, x in x_dict.items():
            mask = node_type == key
            # It will make the training or inference speed slower.
            x_tmp = x[local_node_idx[mask]].to(node_type.device)
            h_tmp = self.lins[str(key)](x_tmp)
            h[mask] = h_tmp.to(device)

        return h

    def forward(self, x_dict, edge_index, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx)

        for layer, conv in enumerate(self.convs):
            x_pre = x
            x = conv(x, edge_index, edge_type)
            if layer != self.num_layers - 1:
                if self.residual:
                    x = x + x_pre
                if self.use_bn:
                    x = self.bns[layer](x)
                x = self.prelus[layer](x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.log_softmax(dim=-1)

        
    def inference(self, x_dict, subgraph_loader, edge_type, node_type, local_node_idx, device):
        x_all = self.group_input(x_dict, node_type, local_node_idx, device=torch.device('cpu'))
        for layer in range(self.num_layers):
            xs = []

            pbar = tqdm(total=num_nodes)
            pbar.set_description(f'Layer {layer:01d}')

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                # print(n_id[:size[1]])
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[layer]((x, x_target), edge_index, edge_type[e_id])
                if layer != self.num_layers - 1:
                    if self.residual:
                        x = x + x_target
                    if self.use_bn:
                        x = self.bns[layer](x)
                    x = self.prelus[layer](x)
                xs.append(x.cpu())
                pbar.update(batch_size)

            pbar.close()
            x_all = torch.cat(xs, dim=0)

        return x_all


device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = REGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers, args.scaling_factor,
             args.dropout, num_feature_dict, len(edge_index_dict.keys()), args.use_bn, args.residual, args.gcn).to(device)

# sum_p = sum(p.numel() for p in model.parameters())
# print(sum_p)

# Create global label vector.
y_global = node_type.new_full((node_type.size(0), 1), -1)
y_global[local2global['paper']] = data.y_dict['paper']

# Move everything to the GPU.
# x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_type = edge_type.to(device)
node_type = node_type.to(device)
local_node_idx = local_node_idx.to(device)
y_global = y_global.to(device)


def train(epoch, optimizer):
    model.train()

    pbar = tqdm(total=args.num_steps * args.train_batch_size)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,
                    data.local_node_idx)
        out = out[data.train_mask]
        y = data.y[data.train_mask].squeeze()
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        num_examples = data.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        pbar.update(args.train_batch_size)

    pbar.close()

    return total_loss / total_examples


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


time_used = []
test()  # Test if inference on GPU succeeds.
for run in range(args.runs):
    st = time.perf_counter()
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        loss = train(epoch, optimizer)
        if epoch % args.test_step == 0:
            result = test()
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}%, '
                f'Test: {100 * test_acc:.2f}%')
    time_used.append(time.perf_counter()-st)
    logger.print_statistics(run)
logger.print_statistics()
time_used = torch.tensor(time_used)
print("time used:", time_used.mean(), time_used.std())