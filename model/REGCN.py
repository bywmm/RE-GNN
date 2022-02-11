import torch
import torch.nn as nn
import dgl

import dgl.function as fn
from layer import REGraphConv


class REGCN(nn.Module):
    def __init__(self,
                 g,
                 num_etypes,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 feats_dim_list):
        super(REGCN, self).__init__()
        self.g = g
        self.num_layers = n_layers
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, in_feats, bias=True) for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(REGraphConv(num_etypes, in_feats, n_hidden, bias=False, activation=None, dropout=dropout, weight=False))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(REGraphConv(num_etypes, n_hidden, n_hidden, activation=activation, dropout=dropout))
        # output layer
        self.layers.append(REGraphConv(num_etypes, n_hidden, n_classes, dropout=dropout))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h = self.layers[0](self.g, h, e_feat)

        for l in range(1, self.num_layers+1):
            h = self.dropout(h)
            h = self.layers[l](self.g, h, e_feat)
        return h
