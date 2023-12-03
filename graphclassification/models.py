import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool

from mlp import MLP
from src.layers import *
from src.solvers import *


class DIGNN_RW(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 mu: float,
                 max_iter: int,
                 num_layers: int,
                 threshold: float,
                 dropout: float,
                 **kwargs):
        super(DIGNN_RW, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = mu
        self.max_iter = max_iter
        self.num_layers = num_layers
        self.threshold = threshold
        self.dropout = dropout
        self.kwargs = kwargs

        self.MLP = MLP(input_dim=in_channels, output_dim=hidden_channels, num_neurons=[64, hidden_channels])
        self.fcs = torch.nn.ModuleList()
        self.graph_fcs = torch.nn.ModuleList()

        self.bn1 = nn.BatchNorm1d(hidden_channels)

        func = ImplicitFunc_RW(hidden_channels, hidden_channels, mu)
        self.model = DEQFixedPoint(func, fwd_solver, fwd_solver, max_iter, threshold)

        for _ in range(num_layers):
            self.fcs.append(nn.Linear(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.graph_fcs.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.final_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.MLP(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn1(x)
        output = self.model(x, edge_index, edge_weight)
        for i in range(self.num_layers):
            output = F.relu(self.fcs[i](output))
            output = F.dropout(output, self.dropout, training=self.training)
        
        output = global_add_pool(output, batch=batch)
        for graph_fc in self.graph_fcs:
            output = F.relu(graph_fc(output))
            output = F.dropout(output, self.dropout, training=self.training)
        output = self.final_out(output)
        output = F.dropout(output, 0.5, training=self.training)
        return F.log_softmax(output, dim=1)
 


class DIGNN_Neural(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 mu: float,
                 max_iter: int,
                 num_layers: int,
                 threshold: float,
                 dropout: float,
                 **kwargs):
        super(DIGNN_Neural, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = mu
        self.max_iter = max_iter
        self.num_layers = num_layers
        self.threshold = threshold
        self.dropout = dropout
        self.kwargs = kwargs

        self.MLP = MLP(input_dim=in_channels, output_dim=hidden_channels, num_neurons=[64, hidden_channels])
        self.fcs = torch.nn.ModuleList()
        self.graph_fcs = torch.nn.ModuleList()

        self.bn1 = nn.BatchNorm1d(hidden_channels)

        func = ImplicitFunc_Neural(hidden_channels, hidden_channels, mu)
        self.model = DEQFixedPoint(func, fwd_solver, fwd_solver, max_iter, threshold)

        for _ in range(num_layers):
            self.fcs.append(nn.Linear(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.graph_fcs.append(nn.Linear(hidden_channels, hidden_channels))
        
        self.final_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.MLP(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn1(x)
        output = self.model(x, edge_index, edge_weight)
        for i in range(self.num_layers):
            output = F.relu(self.fcs[i](output))
            output = F.dropout(output, self.dropout, training=self.training)
        
        output = global_add_pool(output, batch=batch)
        for graph_fc in self.graph_fcs:
            output = F.relu(graph_fc(output))
            output = F.dropout(output, self.dropout, training=self.training)
        output = self.final_out(output)
        output = F.dropout(output, 0.5, training=self.training)
        return F.log_softmax(output, dim=1)
    



        
    
