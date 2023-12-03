import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers import *
from src.solvers import *


class DIGNN_RW(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 mu: float,
                 max_iter: int,
                 threshold: float,
                 dropout: float,
                 num_layers: int=4,
                 **kwargs):
        super(DIGNN_RW, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = mu
        self.max_iter = max_iter
        self.threshold = threshold
        self.dropout = dropout
        self.num_layers = num_layers
        self.kwargs = kwargs

        self.extractor = nn.Linear(in_channels, hidden_channels)
        func = ImplicitFunc_RW(mu)
        self.model = DEQFixedPoint(func, fwd_solver, fwd_solver, max_iter, threshold)
        
        self.res_in_fc = nn.Linear(hidden_channels, hidden_channels)
        self.fc0 = nn.Linear(hidden_channels, hidden_channels)
        self.fcs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.fcs.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.bn_extr = nn.BatchNorm1d(hidden_channels)
        self.bn0 = nn.BatchNorm1d(hidden_channels)
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.final_out = nn.Linear(hidden_channels, out_channels)


    def forward(self, x, edge_index, edge_weight=None):
        x = self.bn_extr(self.extractor(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        output = self.model(x, edge_index, edge_weight)
        output = F.elu(self.bn0(self.fc0(output) + self.res_in_fc(x)))
        output = F.dropout(output, p=self.dropout, training=self.training)

        for i in range(self.num_layers):
            output = F.elu(self.bns[i](self.fcs[i](output) + output))
            output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.final_out(output)
        return output
    

class DIGNN_Neural(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 mu: float,
                 max_iter: int,
                 threshold: float,
                 dropout: float,
                 num_layers: int=4,
                 **kwargs):
        super(DIGNN_Neural, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = mu
        self.max_iter = max_iter
        self.threshold = threshold
        self.dropout = dropout
        self.num_layers = num_layers
        self.kwargs = kwargs

        func = ImplicitFunc_Neural(in_channels, hidden_channels, mu)
        self.model = DEQFixedPoint(func, fwd_solver, fwd_solver, max_iter, threshold)
        self.res_in_fc = nn.Linear(in_channels, hidden_channels)
        self.fc0 = nn.Linear(in_channels, hidden_channels)
        self.fcs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.fcs.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.bn0 = nn.BatchNorm1d(hidden_channels)
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.final_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        output = self.model(x, edge_index, edge_weight)
        output = F.elu(self.bn0(self.fc0(output) + self.res_in_fc(x)))
        output = F.dropout(output, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            output = F.elu(self.bns[i](self.fcs[i](output) + output))
            output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.final_out(output)
        return output