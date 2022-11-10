import argparse
from tqdm import tqdm
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList, Linear, ModuleDict, Parameter, init, ParameterDict
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.nn import MessagePassing, GATv2Conv
from utils import weighted_degree, get_self_loop_index, softmax
from utils import MsgNorm, args_print

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


class REGCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_node_types,
                 num_edge_types,
                 scaling_factor=100.,
                 gcn=False,
                 dropout=0., 
                 use_softmax=False,
                 residual=False,
                 use_norm=None,
                 self_loop_type=1
                ):
        super(REGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_softmax = use_softmax
        self.dropout = dropout
        self.residual = residual
        self.use_norm = use_norm
        self.self_loop_type = self_loop_type

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if self.residual:
            self.weight_root = self.weight # Parameter(torch.Tensor(in_channels, out_channels))
            # self.weight_root = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        if self.self_loop_type in [1, 3]:
            rw_dim = self.num_edge_types
        else:
            rw_dim = self.num_edge_types + num_node_types
        if gcn:
            self.relation_weight = Parameter(torch.Tensor(rw_dim), requires_grad=False)
        else:
            self.relation_weight = Parameter(torch.Tensor(rw_dim), requires_grad=True)
        self.scaling_factor = scaling_factor

        if self.use_norm  == 'bn':
            self.norm = torch.nn.BatchNorm1d(out_channels)
        elif self.use_norm == 'ln':
            self.norm = torch.nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.residual:
            init.xavier_uniform_(self.weight_root)
        init.zeros_(self.bias)
        init.constant_(self.relation_weight, 1.0 / self.scaling_factor)
        if self.use_norm in ['bn', 'ln']:
            self.norm.reset_parameters()

    def forward(self, x, edge_index, edge_type, target_node_type, return_weights=False):
        # shape of x: [N, in_channels]
        # shape of edge_index: [2, E]
        # shape of edge_type: [E]
        # shape of e_feat: [E, edge_tpes+node_types]

        if self.self_loop_type in [1, 3]:
            edge_type = edge_type.view(-1, 1)
            e_feat = torch.zeros(edge_type.shape[0], self.num_edge_types, device=edge_type.device).scatter_(1, edge_type, 1.0)

        elif self.self_loop_type == 2:
            # add self-loops to edge_index and edge_type
            num_nodes = target_node_type.size(0)
            loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)
            edge_type = torch.cat([edge_type, target_node_type+self.num_edge_types], dim=0)

            edge_type = edge_type.view(-1, 1)
            e_feat = torch.zeros(edge_type.shape[0], self.num_edge_types+self.num_node_types, device=edge_type.device).scatter_(1, edge_type, 1.0)

        x_src, x_target = x
        x_src = torch.matmul(x_src, self.weight)
        if self.residual:
            x_target = torch.matmul(x_target, self.weight_root)
        else:
            x_target = torch.matmul(x_target, self.weight)
        x = (x_src, x_target)

        # Cal edge weight according to its relation type
        relation_weight = self.relation_weight * self.scaling_factor
        relation_weight = F.leaky_relu(relation_weight)
        # print(relation_weight)
        edge_weight = torch.matmul(e_feat, relation_weight)  # [E]

        # Compute GCN normalization
        row, col = edge_index

        # self.use_softmax = True
        if self.use_softmax:
            ew = softmax(edge_weight, col)
        else:
            # mean aggregator
            deg = weighted_degree(col, edge_weight, x_target.size(0), dtype=x_target.dtype) #.abs()
            deg_inv = deg.pow(-1.0)
            norm = deg_inv[col]
            ew = edge_weight * norm
        
        # ew = F.dropout(ew, p=self.dropout, training=self.training)
        out = self.propagate(edge_index, x=x, ew=edge_weight)

        if self.residual:
            out += x_target
        
        if self.use_norm in ['bn', 'ln']:
            out = self.norm(out)

        if return_weights:
            return out, ew, relation_weight
        else:
            return out

    def message(self, x_j, ew):

        return ew.view(-1, 1) * x_j

    def update(self, aggr_out):

        aggr_out = aggr_out + self.bias

        return aggr_out


class REGATConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_node_types,
                 num_edge_types,
                 heads=1,
                 scaling_factor=100.,
                 concat=True,
                 negative_slope=0.2,
                 dropout = 0.0,
                 residual=False,
                 use_norm=None,
                 self_loop_type=1):
        super(REGATConv, self).__init__(node_dim=0, aggr='add')
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.residual = residual
        self.use_norm = use_norm
        self.self_loop_type = self_loop_type
        if self.concat:
            self.out_dim = heads * out_channels
        else:
            self.out_dim = out_channels

        self.lin_src = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = self.lin_src
        self.bias = Parameter(torch.Tensor(self.out_dim))
        if self.self_loop_type in [1, 3]:
            rw_dim = self.num_edge_types
        else:
            rw_dim = self.num_edge_types + num_node_types
        
        self.relation_weight = Parameter(torch.Tensor(rw_dim, heads), requires_grad=True)
        self.scaling_factor = scaling_factor

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if self.use_norm  == 'bn':
            self.norm = torch.nn.BatchNorm1d(self.out_dim)
        elif self.use_norm == 'ln':
            self.norm = torch.nn.LayerNorm(self.out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        init.xavier_uniform_(self.att_src)
        init.xavier_uniform_(self.att_dst)
        init.zeros_(self.bias)
        init.constant_(self.relation_weight, 1.0 / self.scaling_factor)
        if self.use_norm in ['bn', 'ln']:
            self.norm.reset_parameters()

    def forward(self, x, edge_index, edge_type, target_node_type, return_weights=False):

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if self.self_loop_type in [1, 3]:
            edge_type = edge_type.view(-1, 1)
            e_feat = torch.zeros(edge_type.shape[0], self.num_edge_types, device=edge_type.device).scatter_(1, edge_type, 1.0)

        elif self.self_loop_type == 2:
            # add self-loops to edge_index and edge_type
            num_nodes = target_node_type.size(0)
            loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)
            edge_type = torch.cat([edge_type, target_node_type+self.num_edge_types], dim=0)

            edge_type = edge_type.view(-1, 1)
            e_feat = torch.zeros(edge_type.shape[0], self.num_edge_types+self.num_node_types, device=edge_type.device).scatter_(1, edge_type, 1.0)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)

        # Cal edge weight according to its relation type
        relation_weight = self.relation_weight * self.scaling_factor
        relation_weight = F.leaky_relu(relation_weight)
        # print(relation_weight)
        edge_weight = torch.matmul(e_feat, relation_weight)  # [E]

        row, col = edge_index
        alpha_i = alpha_src[row]
        alpha_j = alpha_dst[col]
        edge_weight = edge_weight + alpha_i + alpha_j
        edge_weight = F.leaky_relu(edge_weight, self.negative_slope)

        ew = softmax(edge_weight, col)

        # print(x[0].shape, x[1].shape)
        # print(edge_index.shape, ew.shape)
        out = self.propagate(edge_index, x=x, ew=ew)
    
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out += self.bias

        if self.residual:
            out += x_dst.view(-1, self.heads * self.out_channels)
        
        if self.use_norm in ['bn', 'ln']:
            out = self.norm(out)

        if return_weights:
            return out, ew
        else:
            return out

    def message(self, x_j, ew):

        return ew.view(-1, self.heads, 1) * x_j


class REGATv2Conv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_node_types,
                 num_edge_types,
                 heads=1,
                 scaling_factor=100.,
                 concat=True,
                 negative_slope=0.2,
                 dropout = 0.0,
                 residual=False,
                 use_norm=None,
                 self_loop_type=1):
        super(REGATv2Conv, self).__init__(node_dim=0, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.residual = residual
        self.use_norm = use_norm
        self.self_loop_type = self_loop_type
        if self.concat:
            self.out_dim = heads * out_channels
        else:
            self.out_dim = out_channels

        self.lin_src = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = self.lin_src
        self.bias = Parameter(torch.Tensor(self.out_dim))
        if self.self_loop_type in [1, 3]:
            rw_dim = self.num_edge_types
        else:
            rw_dim = self.num_edge_types + num_node_types
        
        self.relation_weight = Parameter(torch.Tensor(rw_dim, heads), requires_grad=True)
        self.scaling_factor = scaling_factor

        # The learnable parameters to compute attention coefficients:
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if self.use_norm  == 'bn':
            self.norm = torch.nn.BatchNorm1d(self.out_dim)
        elif self.use_norm == 'ln':
            self.norm = torch.nn.LayerNorm(self.out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        init.xavier_uniform_(self.att)
        init.zeros_(self.bias)
        init.constant_(self.relation_weight, 1.0 / self.scaling_factor)
        if self.use_norm in ['bn', 'ln']:
            self.norm.reset_parameters()

    def forward(self, x, edge_index, edge_type, target_node_type, return_weights=False):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if self.self_loop_type in [1, 3]:
            edge_type = edge_type.view(-1, 1)
            e_feat = torch.zeros(edge_type.shape[0], self.num_edge_types, device=edge_type.device).scatter_(1, edge_type, 1.0)

        elif self.self_loop_type == 2:
            # add self-loops to edge_index and edge_type
            num_nodes = target_node_type.size(0)
            loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)
            edge_type = torch.cat([edge_type, target_node_type+self.num_edge_types], dim=0)

            edge_type = edge_type.view(-1, 1)
            e_feat = torch.zeros(edge_type.shape[0], self.num_edge_types+self.num_node_types, device=edge_type.device).scatter_(1, edge_type, 1.0)

        row, col = edge_index
        x_i = x_src[row]
        x_j = x_dst[col]
        x_all = x_i + x_j

        x_all = F.leaky_relu(x_all, self.negative_slope)
        alpha = (x_all * self.att).sum(dim=-1)

        # Cal edge weight according to its relation type
        relation_weight = self.relation_weight * self.scaling_factor
        relation_weight = F.leaky_relu(relation_weight)
        # print(relation_weight)
        edge_weight = torch.matmul(e_feat, relation_weight)  # [E]

        edge_weight = edge_weight + alpha
        ew = softmax(edge_weight, col)

        # ew = F.dropout(ew, p=self.dropout, training=self.training)

        # print(edge_index.shape, ew.shape)
        out = self.propagate(edge_index, x=x, ew=ew)
    
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out += self.bias

        if self.residual:
            out += x_dst.view(-1, self.heads * self.out_channels)
        
        if self.use_norm in ['bn', 'ln']:
            out = self.norm(out)

        if return_weights:
            return out, ew
        else:
            return out

    def message(self, x_j, ew):

        return ew.view(-1, self.heads, 1) * x_j