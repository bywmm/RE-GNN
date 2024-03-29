from copy import copy
import argparse
from tqdm import tqdm
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing
from utils import MsgNorm, args_print

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


parser = argparse.ArgumentParser(description='OGBN-MAG (RGCN-NS)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=2048)
parser.add_argument('--regcn_like', action='store_true')
parser.add_argument('--scaling_factor', type=float, default=100.)
parser.add_argument('--gcn_like', action='store_true')
parser.add_argument('--subgraph_test', action='store_true')
parser.add_argument('--Norm4', action='store_true')
parser.add_argument('--use_scheduler', action='store_true')
args = parser.parse_args()
args_print(args)

root = '/irip/wangjunfu_2021/dataset/graph/OGB'
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
subgraph_loader = NeighborSampler(edge_index, node_idx=None,
                               sizes=[-1], batch_size=args.test_batch_size, shuffle=False,
                               num_workers=12)

class RGCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_node_types,
                 num_edge_types,
                 Norm4
                ):
        super(RGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.regcn_like = args.regcn_like
        self.gcn_like = args.gcn_like
        self.Norm4 = False # Norm4

        if self.regcn_like or self.gcn_like:
            self.rel_lins = ModuleList([Linear(in_channels, out_channels, bias=False)])
            self.root_lins = ModuleList([Linear(in_channels, out_channels, bias=True)])
            self.relation_weight = Parameter(torch.Tensor(num_edge_types+num_node_types), requires_grad=True if self.regcn_like else False)
            self.scaling_factor = args.scaling_factor
        else:
            self.rel_lins = ModuleList([
                Linear(in_channels, out_channels, bias=False)
                for _ in range(num_edge_types)
            ])
            self.root_lins = ModuleList([
                Linear(in_channels, out_channels, bias=True)
                for _ in range(num_node_types)
            ])
        if self.Norm4:
            self.msg_norm = ModuleList([MsgNorm(True) for _ in range(num_node_types)])
            self.layer_norm = ModuleList([torch.nn.LayerNorm(out_channels) for _ in range(num_node_types)])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.rel_lins:
            lin.reset_parameters()
        for lin in self.root_lins:
            lin.reset_parameters()
        if self.regcn_like or self.gcn_like:
            torch.nn.init.constant_(self.relation_weight, 1.0 / self.scaling_factor)
        if self.Norm4:
            for n in self.msg_norm:
                n.reset_parameters()
            for n in self.layer_norm:
                n.reset_parameters()

    def forward(self, x, edge_index, edge_type, target_node_type, src_node_type):
        x_src, x_target = x

        if self.regcn_like or self.gcn_like:
            relation_weight = self.relation_weight * self.scaling_factor
            relation_weight = F.leaky_relu(relation_weight)

        out = x_target.new_zeros(x_target.size(0), self.out_channels)

        for i in range(self.num_edge_types):
            mask = edge_type == i
            if self.Norm4:
                out.add_(F.normalize(self.propagate(edge_index[:, mask], x=x, edge_type=i, src_node_type = src_node_type)))
            else:
                out.add_(self.propagate(edge_index[:, mask], x=x, edge_type=i))

        for i in range(self.num_node_types):
            mask = target_node_type == i
            if self.Norm4:
                x = self.root_lins[i](x_target[mask])
                out[mask] = x + self.msg_norm[i](x, out[mask])
                out[mask] = self.layer_norm[i](out[mask])
            else:
                if self.regcn_like or self.gcn_like:
                    out[mask] += relation_weight[i] * self.root_lins[0](x_target[mask])
                else:
                    out[mask] += self.root_lins[i](x_target[mask])
        return out

    def message(self, x_j, edge_type: int):
        if self.regcn_like or self.gcn_like:
            relation_weight = self.relation_weight * self.scaling_factor
            relation_weight = F.leaky_relu(relation_weight)
            return self.rel_lins[0](x_j) * relation_weight[edge_type + self.num_node_types]
        else:
            return self.rel_lins[edge_type](x_j)

class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types, Norm4):
        super(RGCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.Norm4 = Norm4

        self.regcn_like = args.regcn_like
        self.gcn_like = args.gcn_like

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
        self.convs.append(RGCNConv(I, H, num_node_types, num_edge_types, self.Norm4))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(H, H, num_node_types, num_edge_types, self.Norm4))
        self.convs.append(RGCNConv(H, O, num_node_types, num_edge_types, self.Norm4))

        if self.Norm4:
            self.norm = torch.nn.LayerNorm(I)

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()
        if self.Norm4:
            self.norm.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None, device=None):
        # Create global node feature matrix.
        if n_id is not None:
            node_type = node_type[n_id]
            local_node_idx = local_node_idx[n_id]

        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]].to(device)
        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]].to(device)

        return h

    def forward(self, n_id, x_dict, adjs, edge_type, node_type, local_node_idx):

        x = self.group_input(x_dict, node_type, local_node_idx, n_id, node_type.device)

        if self.Norm4:
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.norm(x)

        node_type = node_type[n_id]

        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target node embeddings.
            source_node_type = node_type
            node_type = node_type[:size[1]]  # Target node types.
            conv = self.convs[i]
            x = conv((x, x_target), edge_index, edge_type[e_id], node_type, source_node_type)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_dict, subgraph_loader, edge_type, node_type, local_node_idx, device):
        x_all = self.group_input(x_dict, node_type, local_node_idx, device=torch.device('cpu'))
        if self.Norm4:
            x_all = self.norm(x_all.to(device)).to('cpu')
        for layer in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, e_id, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                node_type_src = node_type[n_id]
                node_type_target = node_type_src[:size[1]]
                x = self.convs[layer]((x, x_target), edge_index, edge_type[e_id], node_type_target, node_type_src)
                if layer != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    def inference_full_batch(self, x_dict, edge_index_dict, key2int):
        # We can perform full-batch inference on GPU.

        device = list(x_dict.values())[0].device

        x_dict = copy(x_dict)
        for key, emb in self.emb_dict.items():
            x_dict[int(key)] = emb

        adj_t_dict = {}
        for key, (row, col) in edge_index_dict.items():
            adj_t_dict[key] = SparseTensor(row=col, col=row).to(device)

        for i, conv in enumerate(self.convs):
            out_dict = {}
            if self.regcn_like or self.gcn_like:
                relation_weight = conv.relation_weight * conv.scaling_factor
                relation_weight = F.leaky_relu(relation_weight)
                print(f"Layer {i}, Relation weight {relation_weight}")

            for j, x in x_dict.items():
                if self.regcn_like or self.gcn_like:
                    out_dict[j] = conv.root_lins[0](x)
                    out_dict[j] *= relation_weight[j]
                else:
                    out_dict[j] = conv.root_lins[j](x)

            for keys, adj_t in adj_t_dict.items():
                src_key, target_key = keys[0], keys[-1]
                out = out_dict[key2int[target_key]]
                tmp = adj_t.matmul(x_dict[key2int[src_key]], reduce='mean')
                if self.regcn_like or self.gcn_like:
                    tmp = conv.rel_lins[0](tmp)
                    tmp *= relation_weight[key2int[keys] + conv.num_node_types]
                    out.add_(tmp)
                else:
                    out.add_(conv.rel_lins[key2int[keys]](tmp))

            if i != self.num_layers - 1:
                for j in range(self.num_node_types):
                    F.relu_(out_dict[j])

            x_dict = out_dict

        return x_dict


device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = RGCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
             args.dropout, num_nodes_dict, list(x_dict.keys()),
             len(edge_index_dict.keys()), args.Norm4).to(device)
# sum_p = sum(p.numel() for p in model.parameters())
# print(sum_p)
# assert False
# Create global label vector.
y_global = node_type.new_full((node_type.size(0), 1), -1)
y_global[local2global['paper']] = data.y_dict['paper']

# Move everything to the GPU.
x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_type = edge_type.to(device)
node_type = node_type.to(device)
local_node_idx = local_node_idx.to(device)
y_global = y_global.to(device)


def train(epoch, optimizer, scheduler, train_steps):
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
        train_steps += 1
        if scheduler is not None:
            scheduler.step(train_steps)
        pbar.update(batch_size)
        # pbar.set_description(f'Vanilla Epoch {epoch:02d}, train acc: {100 * train_acc:.2f}')

    pbar.close()

    loss = total_loss / paper_train_idx.size(0)

    return loss


@torch.no_grad()
def test():
    model.eval()

    # full batch test
    if not args.subgraph_test:
        out = model.inference_full_batch(x_dict, edge_index_dict, key2int)
        out = out[key2int['paper']]
        y_pred = out.argmax(dim=-1, keepdim=True).cpu() # [736389, 1]
        y_true = data.y_dict['paper'] # [736389, 1]
    else:
        out = model.inference(x_dict, subgraph_loader, edge_type, node_type, local_node_idx, device=device)
        y_pred = out.argmax(-1, keepdim=True)[local2global['paper']]
        y_true = y_global[local2global['paper']]
        # pbar = tqdm(total=paper_idx.size(0))
        # pbar.set_description(f'Test')

        # y_pred = []
        # y_true = []
        # for batch_size, n_id, adjs in test_loader:
        #     n_id = n_id.to(device)
        #     adjs = [adj.to(device) for adj in adjs]
        #     out = model(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        #     y_t = y_global[n_id][:batch_size]
        #     y_p = out.argmax(dim=-1, keepdim=True).cpu()
        #     # print(y_t.shape)
        #     # print(y_p.shape)
        #     # print("=====")
        #     y_pred.append(y_p)
        #     y_true.append(y_t)
        #     pbar.update(batch_size)

        # pbar.close()

        # y_pred = torch.cat(y_pred, dim=0)
        # y_true = torch.cat(y_true, dim=0)

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
    steps_per_epoch=len(train_loader)
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * args.epochs, eta_min=args.lr / 100)
    else:
        scheduler = None
    train_steps = 0
    for epoch in range(1, 1 + args.epochs):
        loss = train(epoch, optimizer, scheduler, train_steps)
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