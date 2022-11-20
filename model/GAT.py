import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 n_hidden,
                 n_heads,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 use_sage=False):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = n_layers

        self.layers = nn.ModuleList()
        for i in range(0, n_layers - 1):
            in_c = in_dim if i == 0 else n_hidden * n_heads
            self.layers.append(GATConv(in_c, n_hidden, n_heads, dropout, dropout, activation=activation))
        self.layers.append(GATConv(n_hidden * n_heads, n_classes, 1, dropout, dropout))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h):
        for l in range(0, self.num_layers):
            h = self.layers[l](self.g, h).flatten(1)
            if l != self.num_layers - 1:
                h = self.dropout(h)
        return h, h
