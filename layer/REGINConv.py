import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn import expand_as_pair

class REGINConv(nn.Module):
    def __init__(self,
                 num_etypes,
                 scaling_factor,
                 apply_func=None,
                 aggregator_type='sum',
                 init_eps=0,
                 learn_eps=False,
                 activation=None):
        super(REGINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        if aggregator_type not in ('sum', 'max', 'mean'):
            raise KeyError(
                'Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))
        # may add multi-head
        self.edge_weight = nn.Parameter(th.Tensor(num_etypes, 1), requires_grad=True)
        self.alpha = scaling_factor

        self.reset_parameters()

    def reset_parameters(self):
        self.apply_func.reset_parameters()
        nn.init.constant_(self.edge_weight, 1.0 / self.alpha)

    def forward(self, graph, feat, e_feat):
        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            edge_weight = self.edge_weight * self.alpha
            edge_weight = nn.LeakyReLU()(edge_weight)
            ew = edge_weight[e_feat-1]
            graph.edata.update({'ew': ew})
            
            num_nodes = graph.num_nodes()
            graph.ndata.update({'nones': th.ones(num_nodes, 1).to(feat.device)})
            graph.update_all(fn.u_mul_e('nones', 'ew', 'm'),
                             fn.sum('m', 'norm'))
            norm = th.pow(graph.ndata['norm'].squeeze().clamp(min=1), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src * norm
            graph.update_all(fn.u_mul_e('h', 'ew', 'm'),
                             fn.sum(msg='m', out='h'))
            # rst = (1 + self.eps) * feat_dst + graph.ndata['h']
            rst = graph.ndata['h'] * norm
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            return rst
