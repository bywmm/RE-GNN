import torch
import torch.nn as nn
from layer import REGraphConv, RESAGEConv


class REGCN(nn.Module):
    def __init__(self,
                 g,
                 num_etypes,
                 R,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 feats_dim_list,
                 use_sage=False):
        super(REGCN, self).__init__()
        self.g = g
        self.num_layers = n_layers
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, in_feats, bias=True) for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.layers = nn.ModuleList()
        GConv = RESAGEConv if use_sage else REGraphConv
        self.layers.append(GConv(num_etypes, R, in_feats, n_hidden, bias=False, activation=None, dropout=dropout, weight=False))
        for i in range(1, n_layers - 1):
            self.layers.append(GConv(num_etypes, R, n_hidden, n_hidden, activation=activation, dropout=dropout))
        self.layers.append(GConv(num_etypes, R, n_hidden, n_classes, bias=False, dropout=dropout, weight=False))
        self.out_lin = nn.Linear(n_hidden, n_classes, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h = self.layers[0](self.g, h, e_feat)

        for l in range(1, self.num_layers):
            h = self.dropout(h)
            h = self.layers[l](self.g, h, e_feat)
        out = self.out_lin(h)
        return out, h
