import torch
import torch.nn as nn

from layer import REGATConv, REGATv2Conv

class REGAT(nn.Module):
    def __init__(self,
                 g,
                 num_etypes,
                 R,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 feats_dim_list,
                 use_gatv2=False):
        super(REGAT, self).__init__()
        self.g = g
        self.num_etypes = num_etypes
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.feats_dim_list = feats_dim_list
        self.activation = activation

        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, num_hidden, bias=True) for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        GConv = REGATv2Conv if use_gatv2 else REGATConv
        # input projection (no residual)
        self.gat_layers.append(GConv(
            self.num_etypes, R, in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers-1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            # self.bns.append(nn.BatchNorm1d(num_hidden * heads[l-1]))
            self.gat_layers.append(GConv(
                self.num_etypes, R, num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        self.bns.append(nn.BatchNorm1d(num_hidden * heads[-2]))
        # output projection
        self.gat_layers.append(GConv(
            self.num_etypes, R, num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h = self.gat_layers[0](self.g, h, e_feat).flatten(1)
        
        for l in range(1, self.num_layers-1):
            h = self.gat_layers[l](self.g, h, e_feat).flatten(1)
            # h = self.bns[l](h)
        # output projection
        logits = self.gat_layers[-1](self.g, h, e_feat).mean(1)
        return logits
