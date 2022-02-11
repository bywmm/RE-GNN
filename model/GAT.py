import torch
import torch.nn as nn

from dgl.nn.pytorch import edge_softmax, GATConv
from layer import APLayer


class GAT(nn.Module):
    def __init__(self,
                 g,
                 feats_type,
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
                 feats_dim_list):
        super(GAT, self).__init__()
        self.g = g
        self.feats_type = feats_type
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if self.feats_type in [0, 1, 2, 3]:
            self.fc_list = nn.ModuleList([nn.Linear(feats_dim, in_dim, bias=True) for feats_dim in feats_dim_list])
            for fc in self.fc_list:
                nn.init.xavier_normal_(fc.weight, gain=1.414)
        else:
            self.ap_layer = APLayer(feats_dim_list[0], feats_type)
            self.ap_fc = nn.Linear(feats_dim_list[0], in_dim, bias=True)

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

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
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
