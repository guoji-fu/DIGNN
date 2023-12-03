import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv, GATConv, SGConv, APPNP, JumpingKnowledge, GCN2Conv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, to_scipy_sparse_matrix

import scipy.sparse as sp
from scipy.sparse import csr_matrix
from src.normalization import fetch_normalization
from src.layers_baselines import *
from src.utils import *


class MLPNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=16,
                 dropout=0.5):
        super(MLPNet, self).__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, num_hid)
        self.layer2 = torch.nn.Linear(num_hid, out_channels)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.relu(self.layer1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)


class GCNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=16,
                 dropout=0.5,
                 cached=True):
        super(GCNNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid, cached=cached)
        self.conv2 = GCNConv(num_hid, out_channels, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class SGCNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 K=2,
                 cached=True):
        super(SGCNet, self).__init__()
        self.conv1 = SGConv(in_channels, out_channels, K=K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=8,
                 num_heads=8,
                 dropout=0.6,
                 concat=False):

        super(GATNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, num_hid, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(num_heads * num_hid, out_channels, heads=1, concat=concat, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=-1)


class JKNet(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0,
                 num_layes=4,
                 dropout=0.5):
        super(JKNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid)
        self.conv2 = GCNConv(num_hid, num_hid)
        self.lin1 = torch.nn.Linear(num_hid, out_channels)
        self.one_step = APPNP(K=K, alpha=alpha)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=num_hid,
                                   num_layers=num_layes)

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index, edge_weight)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)


class APPNPNet(torch.nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0.1,
                 dropout=0.5):
        super(APPNPNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GCNIINet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 hidden_channels=64, 
                 num_layers=64, 
                 dropout=0.5,
                 alpha=0.5, 
                 theta=1.):
        super(GCNIINet, self).__init__()
        self.dropout = dropout

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCN2Conv(channels=hidden_channels,
                                       alpha=alpha, theta=theta, layer=i+1))
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x_0 = x
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x_0, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x)
        return F.log_softmax(out, dim=1)


class H2GCN_Prop(MessagePassing):
    def __init__(self):
        super(H2GCN_Prop, self).__init__()

    def forward(self, h, norm_adj_1hop, norm_adj_2hop):
        h_1 = torch.sparse.mm(norm_adj_1hop, h) # if OOM, consider using torch-sparse
        h_2 = torch.sparse.mm(norm_adj_2hop, h)
        h = torch.cat((h_1, h_2), dim=1)
        return h


class H2GCNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 hidden_channels, 
                 edge_index, 
                 num_layers=1,
                 dropout=0.5, 
                 act='relu'):
        super(H2GCNNet, self).__init__()
        self.dropout = dropout

        self.lin1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.act = torch.nn.ReLU() if act == 'relu' else torch.nn.Identity()
        self.H2GCN_layer = H2GCN_Prop()
        self.num_layers = num_layers
        self.lin_final = nn.Linear((2**(self.num_layers+1)-1)*hidden_channels, out_channels, bias=False)

        adj = to_scipy_sparse_matrix(remove_self_loops(edge_index)[0])
        adj_2hop = adj.dot(adj)
        adj_2hop = adj_2hop - sp.diags(adj_2hop.diagonal())
        adj = indicator_adj(adj)
        adj_2hop = indicator_adj(adj_2hop)
        norm_adj_1hop = get_normalized_adj(adj)
        self.norm_adj_1hop = sparse_mx_to_torch_sparse_tensor(norm_adj_1hop, 'cuda')
        norm_adj_2hop = get_normalized_adj(adj_2hop)
        self.norm_adj_2hop = sparse_mx_to_torch_sparse_tensor(norm_adj_2hop, 'cuda')

    def forward(self, x, edge_index=None, edge_weight=None):
        hidden_hs = []
        h = self.act(self.lin1(x))
        hidden_hs.append(h)
        for i in range(self.num_layers):
            h = self.H2GCN_layer(h, self.norm_adj_1hop, self.norm_adj_2hop)
            hidden_hs.append(h)
        h_final = torch.cat(hidden_hs, dim=1)
        # print(f'lin_final.size(): {self.lin_final.weight.size()}, h_final.size(): {h_final.size()}')
        h_final = F.dropout(h_final, p=self.dropout, training=self.training)
        output = self.lin_final(h_final)
        return F.log_softmax(output, dim=1)



class IGNNNet(nn.Module):
    def __init__(self, 
                 nfeat, 
                 nhid, 
                 nclass, 
                 num_node, 
                 dropout, 
                 kappa=0.9, 
                 adj_orig=None):
        super(IGNNNet, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        #one layer with V
        self.ig1 = ImplicitGraph(nfeat, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = Parameter(torch.zeros(nhid, num_node), requires_grad=False)
        self.V = nn.Linear(nhid, nclass, bias=False)

    def get_Z_star(self, features, adj):
        x = features
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        return x, F.normalize(x, dim=-1)

    def forward(self, features, adj):
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)

        x = features
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        x = F.normalize(x, dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V(x)
        return x


class IDM_SGC_Linear(nn.Module):
    def __init__(self, 
                 adj, 
                 sp_adj, 
                 m, 
                 m_y, 
                 num_eigenvec, 
                 gamma, 
                 all_eigenvec=True):
        super(IDM_SGC_Linear, self).__init__()
        self.B = nn.Parameter(1. / np.sqrt(m) * torch.randn(m_y, m), requires_grad=True)
        if all_eigenvec:
            self.IDM_SGC = IDM_SGC(adj, sp_adj, m, num_eigenvec, gamma)

    def reset_parameters(self):
        self.IDM_SGC.reset_parameters()

    def forward(self, X, adj=None):
        # TODO
        output = self.IDM_SGC(X)
        output = self.B @ output
        return output.t()


epsilon_F = 10**(-12)
def g(F):
    FF = F.t() @ F
    FF_norm = torch.norm(FF, p='fro')
    return (1/(FF_norm+epsilon_F)) * FF


def get_G(Lambda_F, Lambda_S, gamma):
    G = 1.0 - gamma * Lambda_F @ Lambda_S.t()
    G = 1 / G
    return G
    

class EIGNN_w_iterative(nn.Module):
    def __init__(self, 
                 adj, 
                 sp_adj, 
                 m, 
                 m_y, 
                 threshold, 
                 max_iter, 
                 gamma, 
                 chain_len, 
                 adaptive_gamma=False,
                 spectral_radius_mode=False, 
                 compute_jac_loss=False):
        super(EIGNN_w_iterative, self).__init__()
        # self.B = nn.Parameter(torch.FloatTensor(m_y, m), requires_grad=True)
        if not adaptive_gamma:
            self.EIGNN = EIGNN_w_iterative_solvers(adj, sp_adj, m, threshold, max_iter, gamma,
                                                   spectral_radius_mode=spectral_radius_mode, compute_jac_loss=compute_jac_loss)
        else:
            self.EIGNN = EIGNN_w_iter_adap_gamma(adj, sp_adj, m, threshold, max_iter, gamma, chain_len)
        self.B = nn.Linear(m, m_y, bias=False)
        # self.F = nn.Parameter(torch.FloatTensor(m, m), requires_grad=True)
        # S_dense = S.to_dense()
        # self.S = S
        # self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float), requires_grad=False)
        # self.Lambda_S, self.Q_S = torch.symeig(S_dense, eigenvectors=True)
        # self.Lambda_S = self.Lambda_S.view(-1,1)
        #
        # del S_dense
        self.reset_parameters()

    def reset_parameters(self):
        self.B.reset_parameters()
        self.EIGNN.reset_parameters()
        # torch.nn.init.xavier_uniform_(self.F)

    def forward(self, X):
        output, jac_loss = self.EIGNN(X)
        output = output.t()
        output = F.normalize(output, dim=-1)
        output = F.dropout(output, 0.5, training=self.training)
        output = self.B(output)
        # if self.training:
        #     print(f'gamma: {self.EIGNN.gamma}')
        return output, jac_loss
    