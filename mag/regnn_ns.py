from copy import copy
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter, init
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
from utils import weighted_degree, get_self_loop_index, softmax

# from torchstat import stat

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

parser = argparse.ArgumentParser(description='OGBN-MAG (SAGE)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=2048)
parser.add_argument('--scaling_factor', type=float, default=100.)
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

# Add Self-Loop Relation
self_loop_index = get_self_loop_index(num_node=data.num_nodes_dict['author'])
edge_index_dict[('author', 'selfloop', 'author')] = self_loop_index

self_loop_index = get_self_loop_index(num_node=data.num_nodes_dict['field_of_study'])
edge_index_dict[('field_of_study', 'selfloop', 'field_of_study')] = self_loop_index

self_loop_index = get_self_loop_index(num_node=data.num_nodes_dict['institution'])
edge_index_dict[('institution', 'selfloop', 'institution')] = self_loop_index

self_loop_index = get_self_loop_index(num_node=data.num_nodes_dict['paper'])
edge_index_dict[('paper', 'selfloop', 'paper')] = self_loop_index


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
x_dict = {}
for key, x in data.x_dict.items():
    x_dict[key2int[key]] = x

num_nodes_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N

# Next, we create a train sampler that only iterates over the respective
# paper training nodes.
paper_idx = local2global['paper']
paper_train_idx = paper_idx[split_idx['train']['paper']]

train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=[25, 20], batch_size=args.train_batch_size, shuffle=True,
                               num_workers=12)
test_loader = NeighborSampler(edge_index, node_idx=paper_idx,
                               sizes=[25, 20], batch_size=args.test_batch_size, shuffle=False,
                               num_workers=12)

class REGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types, scaling_factor=100., dropout=0., 
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

        self.relation_weight = Parameter(torch.Tensor(num_edge_types), requires_grad=True)
        # self.self_loop_weight = Parameter(torch.Tensor(num_node_types), requires_grad=True)
        self.scaling_factor = scaling_factor

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)
        init.constant_(self.relation_weight, 1.0 / self.scaling_factor)
        # init.constant_(self.self_loop_weight, 1.0 / self.scaling_factor)

    # def forward(self, x, edge_index, edge_type, target_node_type):
    def forward(self, x, edge_index, edge_type, return_weights=False):
        # shape of x: [N, in_channels]
        # shape of edge_index: [2, E]
        # shape of edge_type: [E]
        # shape of e_feat: [E, edge_tpes+node_types]

        edge_type = edge_type.view(-1, 1)
        e_feat = torch.zeros(edge_type.shape[0], self.num_edge_types, device=edge_type.device).scatter_(1, edge_type.view(-1, 1), 1.0)
        
        x_src, x_target = x
        x_src = torch.matmul(x_src, self.weight)
        x_target = torch.matmul(x_target, self.weight_root)
        x = (x_src, x_target)

        # Cal edge weight according to its relation type
        # relation_weight = torch.cat([self.relation_weight, self.self_loop_weight], dim=-1)
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
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(REGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        num_node_types = len(node_types)

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(REGCNConv(I, H, num_node_types, num_edge_types, scaling_factor))
        for _ in range(num_layers - 2):
            self.convs.append(REGCNConv(H, H, num_node_types, num_edge_types, scaling_factor))
        self.convs.append(REGCNConv(H, O, self.num_node_types, num_edge_types, scaling_factor))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None):
        # Create global node feature matrix.
        if n_id is not None:
            node_type = node_type[n_id]
            local_node_idx = local_node_idx[n_id]

        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, n_id, x_dict, adjs, edge_type, node_type,
                local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx, n_id)
        node_type = node_type[n_id]

        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target node embeddings.
            node_type = node_type[:size[1]]  # Target node types.
            conv = self.convs[i]
            x = conv((x, x_target), edge_index, edge_type[e_id])
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.log_softmax(dim=-1)



device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = REGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers, args.scaling_factor,
             args.dropout, num_nodes_dict, list(x_dict.keys()),
             len(edge_index_dict.keys())).to(device)

# Create global label vector.
y_global = node_type.new_full((node_type.size(0), 1), -1)
y_global[local2global['paper']] = data.y_dict['paper']

# Move everything to the GPU.
x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_type = edge_type.to(device)
node_type = node_type.to(device)
local_node_idx = local_node_idx.to(device)
y_global = y_global.to(device)


def train(epoch, optimizer):
    model.train()

    pbar = tqdm(total=paper_train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

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
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / paper_train_idx.size(0)

    return loss


@torch.no_grad()
def test():
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

time_used = []
# test()  # Test if inference on GPU succeeds.
for run in range(args.runs):
    st = time.perf_counter()
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        loss = train(epoch, optimizer)
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