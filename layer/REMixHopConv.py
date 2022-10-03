import torch
import torch.nn as nn
import dgl.function as fn
from torch.nn import init


class REMixHopConv(nn.Module):
    def __init__(self,
                 num_etypes,
                 scaling_factor,
                 in_feats,
                 out_feats,
                 p=[0, 1, 2],
                 dropout=0,
                 activation=None,
                 batchnorm=False):
        super(REMixHopConv, self).__init__()
        self.in_dim = in_feats
        self.out_dim = out_feats
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # may add multi-head
        self.edge_weight = nn.Parameter(torch.Tensor(num_etypes, 1), requires_grad=True)
        self.alpha = scaling_factor

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_feats * len(p))

        # define weight dict for each power j
        self.weights = nn.ModuleDict(
            {str(j): nn.Linear(in_feats, out_feats, bias=False) for j in p}
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.batchnorm:
            self.bn.reset_parameters()
        for j in self.p:
            self.weights[str(j)].reset_parameters()
        init.constant_(self.edge_weight, 1.0 / self.alpha)

    def forward(self, graph, feats, e_feat):
        with graph.local_scope():
            edge_weight = self.edge_weight * self.alpha
            # edge_weight[6:] = 1.0
            edge_weight = nn.LeakyReLU()(edge_weight)
            ew = edge_weight[e_feat-1]
            # ew = self.ew_dropout(ew)
            graph.edata.update({'ew': ew})
            # print(self.edge_weight, self.weight.shape)

            num_nodes = graph.num_nodes()
            graph.ndata.update({'nones': torch.ones(num_nodes, 1).to(feats.device)})
            graph.update_all(fn.u_mul_e('nones', 'ew', 'm'),
                             fn.sum('m', 'norm'))
            norm = torch.pow(graph.ndata['norm'].squeeze().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feats.dim() - 1)
            norm = torch.reshape(norm, shp).to(feats.device)
            # feat = feat * norm

            # # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            # degs = graph.in_degrees().float().clamp(min=1)
            # norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):

                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)

                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata.pop("h")
                feats = feats * norm

            final = torch.cat(outputs, dim=1)

            if self.batchnorm:
                final = self.bn(final)

            if self.activation is not None:
                final = self.activation(final)

            final = self.dropout(final)

            return final