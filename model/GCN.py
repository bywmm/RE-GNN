import torch
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from layer import APLayer


class GCN(nn.Module):
    def __init__(self,
                 g,
                 feats_type,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 feats_dim_list):
        super(GCN, self).__init__()
        self.g = g
        self.feats_type = feats_type
        self.layers = nn.ModuleList()
        if self.feats_type in [0, 1, 2, 3]:
            self.fc_list = nn.ModuleList([nn.Linear(feats_dim, n_hidden, bias=True) for feats_dim in feats_dim_list])
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)
        else:
            self.ap_layer = APLayer(0, feats_dim_list[0], self.feats_type)
            self.ap_fc = nn.Linear(feats_dim_list[0], n_hidden, bias=True)
        # input layer
        self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, weight=False))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list):
        h = []
        if self.feats_type in [0, 1, 2, 3]:
            for fc, feature in zip(self.fc_list, features_list):
                h.append(fc(feature))
            h = torch.cat(h, 0)
        else:
            for feature in features_list:
                h.append(feature)
            h = torch.cat(h, 0)
            h = self.ap_layer(self.g, h)
            h = self.ap_fc(h)
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(self.g, h)
        return h
