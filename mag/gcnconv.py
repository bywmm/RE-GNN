from torch_geometric.nn.conv import MessagePassing, SAGEConv, GraphConv
from torch_geometric.utils import add_self_loops, degree
import torch
import math
from torch.nn import Linear
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter_add


class indGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(indGCNConv, self).__init__(aggr="mean")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        # shape of x: [N, in_channels]
        # shape of edge_index: [2, E]
        if torch.is_tensor(x):
            x = (x, x)

        out = self.propagate(edge_index, x=x, norm=None)
        out = self.lin(out)

        return out