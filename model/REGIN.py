import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import REGINConv


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
    
    def reset_parameters(self):
        for lin in self.linears:
            lin.reset_parameters()

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class REGIN(nn.Module):
    def __init__(self,
                 g,
                 num_etypes,
                 R,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 activation,
                 dropout,
                 feats_dim_list):
        super().__init__()
        self.g = g
        self.num_layers = n_layers
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, input_dim, bias=True) for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.layers = nn.ModuleList()
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(n_layers):  # excluding the input layer
            in_c = input_dim if layer ==0 else hidden_dim
            out_c = output_dim if layer == n_layers-1 else hidden_dim
            mlp = MLP(in_c, hidden_dim, out_c, activation)
            self.layers.append(REGINConv(num_etypes, R, mlp, learn_eps=False))  

        self.dropout = nn.Dropout(dropout)


    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h = self.layers[0](self.g, h, e_feat)

        for l in range(1, self.num_layers):
            h = self.dropout(h)
            h = self.layers[l](self.g, h, e_feat)
        return h

