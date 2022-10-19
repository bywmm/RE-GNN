from copy import copy
import argparse
from tqdm import tqdm
import time

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger

parser = argparse.ArgumentParser(description='OGBN-MAG (GraphSAINT)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--walk_length', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=30)
args = parser.parse_args()
print(args)

root = '/home/wangjunfu/dataset/graph/OGB'
dataset = PygNodePropPredDataset(root=root, name='ogbn-mag')
real_data = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
logger = Logger(args.runs, args)

# We do not consider those attributes for now.
real_data.node_year_dict = None
real_data.edge_reltype_dict = None

print(real_data)

edge_index_dict = real_data.edge_index_dict

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
out = group_hetero_graph(real_data.edge_index_dict, real_data.num_nodes_dict)
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

data = Data(edge_index=edge_index, edge_attr=edge_type,
                 node_type=node_type, local_node_idx=local_node_idx,
                 num_nodes=node_type.size(0))
print(node_type)
print(node_type.shape)

data.y = node_type.new_full((node_type.size(0), 1), -1)
data.y[local2global['paper']] = real_data.y_dict['paper']

data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
data.train_mask[local2global['paper'][split_idx['train']['paper']]] = True

print(data)

train_loader = GraphSAINTRandomWalkSampler(data,
                                           batch_size=args.batch_size,
                                           walk_length=args.num_layers,
                                           num_steps=args.num_steps,
                                           sample_coverage=0,
                                           save_dir=dataset.processed_dir)


# Map informations to their canonical type.
x_dict = {}
for key, x in real_data.x_dict.items():
    x_dict[key2int[key]] = x

num_nodes_dict = {}
for key, N in real_data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_nodes_dict, x_types, num_edge_types):
        super(GCN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        node_types = list(num_nodes_dict.keys())
        # num_node_types = len(node_types)

        # self.num_node_types = num_node_types
        # self.num_edge_types = num_edge_types

        # Create embeddings for all node types that do not come with features.
        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        I, H, O = in_channels, hidden_channels, out_channels  # noqa

        # Create `num_layers` many message passing layers.
        self.convs = ModuleList()
        self.convs.append(GCNConv(I, H))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(H, H))
        self.convs.append(GCNConv(H, O))

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.emb_dict.values():
            torch.nn.init.xavier_uniform_(emb)
        for conv in self.convs:
            conv.reset_parameters()

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

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

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

model = GCN(128, args.hidden_channels, dataset.num_classes, args.num_layers,
             args.dropout, num_nodes_dict, list(x_dict.keys()),
             len(edge_index_dict.keys())).to(device)

x_dict = {k: v.to(device) for k, v in x_dict.items()}


def train(epoch):
    model.train()

    pbar = tqdm(total=args.num_steps * args.batch_size)
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
        pbar.update(args.batch_size)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    out = []
    cnt = 0
    for data in test_loader:
        cnt += 1
        data = data.to(device)
        per_out = model(x_dict, data.edge_index, data.edge_attr, data.node_type,
                       data.local_node_idx)
        print(cnt, per_out.shape)
        out.append(per_out)
    out = torch.cat(out, dim=0)
    print(out.shape)
    out = out[data.node_type == key2int['paper']]
    print(out.shape)
    print(">>>")
    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    y_true = real_data.y_dict['paper']
    print(y_pred.shape)
    print(y_true.shape)

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
    for epoch in range(1, args.epochs + 1):
        loss = train(epoch)
        torch.cuda.empty_cache()
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
