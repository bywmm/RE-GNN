import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax


class APLayer(nn.Module):
    def __init__(self,
                 num_etypes,
                 in_feats,
                 feats_type,
                 activation=None):
        super(APLayer, self).__init__()
        self.feats_type = feats_type
        if self.feats_type == 4:
            self.attn = nn.Parameter(th.FloatTensor(size=(1, in_feats)))
        elif self.feats_type == 6:
            self.attn = nn.Parameter(th.FloatTensor(size=(1, in_feats)))
            self.edge_weight = nn.Parameter(th.FloatTensor(size=(num_etypes, num_etypes)))
            self.edge_weight_param = nn.Parameter(th.FloatTensor(size=(num_etypes, 1)))
        # self.attn_drop = nn.Dropout(attn_drop)
        self.temperature = 1.0
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.feats_type == 4:
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(self.attn, gain=gain)
        elif self.feats_type == 6:
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(self.attn, gain=gain)
            nn.init.xavier_normal_(self.edge_weight, gain=gain)
            nn.init.xavier_normal_(self.edge_weight_param, gain=gain)

    def forward(self, graph, feat, e_feat=None):
        graph = graph.local_var()
        node_abs = feat.abs().sum(dim=-1).unsqueeze(-1)
        node_mask = th.ones_like(node_abs)
        node_mask[node_abs == 0] = 0
        # return node_mask
        if self.feats_type == 4:
            nw = (feat * self.attn).sum(dim=-1).unsqueeze(-1)
        elif self.feats_type == 5:
            nw = th.pow(graph.in_degrees().float().clamp(min=1), -1.0).unsqueeze(-1)
        elif self.feats_type == 6:
            nw = (feat * self.attn).sum(dim=-1).unsqueeze(-1)
            edge_weight = th.matmul(self.edge_weight, self.edge_weight_param) + 1.0
            # edge_weight = nn.LeakyReLU()(edge_weight)
            # print(edge_weight)
            ew = edge_weight[e_feat - 1]
        else:
            raise NotImplementedError

        # softmax without
        nw_exp = th.exp(nw / self.temperature)
        # only attributed node can propagate message
        nw_exp = nw_exp * node_mask
        graph.ndata.update({'ft': feat, 'nw_exp': nw_exp})

        # compute edge attention
        graph.update_all(fn.copy_src('nw_exp', 'm'),
                         fn.sum('m', 'nw_exp_sum'))

        graph.apply_edges(fn.copy_src('nw_exp', 'e'))
        if self.feats_type == 6:
            ew = graph.edata['e'] * ew
        else:
            ew = graph.edata['e']
        graph.edata.update({'ew': ew})

        graph.update_all(fn.u_mul_e('ft', 'ew', 'm'),
                         fn.sum('m', 'ft'))

        graph.ndata['nw_exp_sum'][graph.ndata['nw_exp_sum'] < 1e-8] = 1
        rst = graph.ndata['ft'] / graph.ndata['nw_exp_sum']


        # only non-attributed node need update message
        rst = rst * (1-node_mask) + feat * node_mask

        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst
