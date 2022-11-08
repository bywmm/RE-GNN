""" The main file to train a MixHop model using a full graph """

import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import dgl
import dgl.function as fn
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from layer import REMixHopConv


class REMixHop(nn.Module):
    def __init__(
        self,
        g,
        num_etypes,
        R,
        in_dim,
        hid_dim,
        out_dim,
        num_layers,
        feats_dim_list,
        p=[0, 1, 2],
        input_dropout=0.0,
        layer_dropout=0.0,
        activation=None,
        batchnorm=False,
    ):
        super(REMixHop, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, in_dim, bias=True) for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # Input layer
        self.layers.append(
            REMixHopConv(
                num_etypes,
                R,
                self.in_dim,
                self.hid_dim,
                p=self.p,
                dropout=self.input_dropout,
                activation=self.activation,
                batchnorm=self.batchnorm,
            )
        )

        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 1):
            self.layers.append(
                REMixHopConv(
                    num_etypes,
                    R,
                    self.hid_dim * len(self.p),
                    self.hid_dim,
                    p=self.p,
                    dropout=self.layer_dropout,
                    activation=self.activation,
                    batchnorm=self.batchnorm,
                )
            )

        self.fc_layers = nn.Linear(
            self.hid_dim * len(self.p), self.out_dim, bias=False
        )

    def forward(self, features_list, e_feat):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        # feats = self.dropout(feats)
        h = self.layers[0](self.g, h, e_feat)

        for l in range(1, self.num_layers):
            h = self.dropout(h)
            h = self.layers[l](self.g, h, e_feat)
        out = self.fc_layers(h)

        return out, h

