import torch as th
from torch import nn
from torch.nn import init

from dgl import function as fn


class RESAGEConv(nn.Module):
    def __init__(self,
                 num_etypes,
                 scaling_factor,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None,
                 weight=True,
                 dropout=0.):
        super(RESAGEConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm = norm
        self.dropout = dropout

        # may add multi-head
        self.edge_weight = nn.Parameter(th.Tensor(num_etypes, 1), requires_grad=True)
        self.alpha = scaling_factor
        
        if weight:
            self.weight_root = nn.Parameter(th.Tensor(in_feats, out_feats))
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight_root', None)
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.feat_dropout = nn.Dropout(p=self.dropout)

        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        
        init.constant_(self.edge_weight, 1.0 / self.alpha)

    def forward(self, graph, feat, e_feat):

        graph = graph.local_var()

        feat = self.feat_dropout(feat)
        if self.weight_root is not None:
            feat_root = th.matmul(feat, self.weight)
        else:
            feat_root = feat
        edge_weight = self.edge_weight * self.alpha
        # edge_weight[6:] = 1.0
        edge_weight = nn.LeakyReLU()(edge_weight)
        ew = edge_weight[e_feat-1]
        # ew = self.ew_dropout(ew)
        graph.edata.update({'ew': ew})
        # print(self.edge_weight, self.weight.shape)

        if self.norm:
            num_nodes = graph.num_nodes()
            graph.ndata.update({'nones': th.ones(num_nodes, 1).to(feat.device)})
            graph.update_all(fn.u_mul_e('nones', 'ew', 'm'),
            # graph.update_all(fn.copy_src(src='nones', out='m'),
                             fn.sum('m', 'norm'))
            # norm = th.pow(graph.ndata['norm'].squeeze().clamp(min=1), -0.5)
            norm = th.pow(graph.ndata['norm'].squeeze().clamp(min=1), -1.0)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self.in_feats > self.out_feats:
            # mult W first to reduce the feature size for aggregation.
            if self.weight is not None:
                feat = th.matmul(feat, self.weight)
            graph.ndata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            graph.update_all(fn.u_mul_e('h', 'ew', 'm'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            graph.update_all(fn.u_mul_e('h', 'ew', 'm'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            if self.weight is not None:
                rst = th.matmul(rst, self.weight)

        # if self.norm:
        #     rst = rst * norm
        
        rst += feat_root

        if self.bias is not None:
            rst = rst + self.bias

        if self.activation is not None:
            rst = self.activation(rst)

        return rst

