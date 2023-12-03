import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import csr_matrix

from torch_geometric.utils import remove_self_loops, to_scipy_sparse_matrix, from_scipy_sparse_matrix

from src.layers import *
from src.solvers import *
from src.utils import *


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x



class DIGNN_RW(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 edge_index,
                 num_nodes,
                 device,
                 mu: float,
                 max_iter: int,
                 threshold: float,
                 dropout: float,
                 preprocess: str='linkx',
                 **kwargs):
        super(DIGNN_RW, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = mu
        self.max_iter = max_iter
        self.threshold = threshold
        self.dropout = dropout
        self.preprocess = preprocess
        self.kwargs = kwargs

        self.fc1 = nn.Linear(in_channels, hidden_channels)

        if preprocess != '':
            row, col = edge_index
            row = row - row.min()
            adj = torch_sparse.SparseTensor(row=row, col=col,
                         sparse_sizes=(num_nodes, num_nodes)
                         ).to_torch_sparse_coo_tensor()
            self.adj = adj.to(device)
        if preprocess == 'linkx':
            self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, num_layers=1, dropout=0.5)
            self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, num_layers=1, dropout=0.5)
            self.W = MLP(2*hidden_channels, 2*hidden_channels, hidden_channels, num_layers=2, dropout=0.5)
            self.fc1 = nn.Linear(hidden_channels, hidden_channels)

        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        func = ImplicitFunc_RW(mu)
        self.model = DEQFixedPoint(func, fwd_solver, fwd_solver, max_iter, threshold)


    def forward(self, x, edge_index, edge_weight=None):
        if self.preprocess == 'adj':
            x = torch.matmul(self.adj, x)
        elif self.preprocess == 'linkx':
            x = F.dropout(x, p=self.dropout, training=self.training)
            xA = self.mlpA(self.adj)
            xX = self.mlpX(x)
            x = torch.cat((xA, xX), axis=-1)
            x = F.relu(self.W(x))
            x = x + 0.5*xA + 0.5*xX
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn1(x)
        x = self.model(x, edge_index, edge_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class DIGNN_Neural(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 edge_index,
                 num_nodes,
                 device,
                 mu: float,
                 max_iter: int,
                 threshold: float,
                 dropout: float,
                 preprocess: str='linkx',
                 **kwargs):
        super(DIGNN_Neural, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = mu
        self.max_iter = max_iter
        self.threshold = threshold
        self.dropout = dropout
        self.preprocess = preprocess
        self.kwargs = kwargs

        self.fc1 = nn.Linear(in_channels, hidden_channels)

        if preprocess != '':
            row, col = edge_index
            row = row - row.min()
            adj = torch_sparse.SparseTensor(row=row, col=col,
                         sparse_sizes=(num_nodes, num_nodes)
                         ).to_torch_sparse_coo_tensor()
            self.adj = adj.to(device)
        if preprocess == 'linkx':
            self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, num_layers=1, dropout=0.5)
            self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, num_layers=1, dropout=0.5)
            self.W = MLP(2*hidden_channels, 2*hidden_channels, hidden_channels, num_layers=2, dropout=0.5)
            self.fc1 = nn.Linear(hidden_channels, hidden_channels)


        self.fc2 = nn.Linear(hidden_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        func = ImplicitFunc_Neural(hidden_channels, hidden_channels, mu)
        self.model = DEQFixedPoint(func, fwd_solver, fwd_solver, max_iter, threshold)

    def forward(self, x, edge_index, edge_weight=None):
        if self.preprocess == 'adj':
            x = torch.matmul(self.adj, x)
        elif self.preprocess == 'linkx':
            x = F.dropout(x, p=self.dropout, training=self.training)
            xA = self.mlpA(self.adj)
            xX = self.mlpX(x)
            x = torch.cat((xA, xX), axis=-1)
            x = F.relu(self.W(x))
            x = x + 0.5*xA + 0.5*xX
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn1(x)
        x = self.model(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)