import os
import sys
import pickle as pkl

import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.datasets import WebKB, WikipediaNetwork
import torch_geometric.transforms as T
from torch_sparse import coalesce

import linkx_dataset


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def get_data(root, name, device):
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_citation(root, name, device)
    if name.lower() in ['chameleon', 'squirrel']:
        return get_heterophilic_dataset(root, name, device)
    if name.lower() in ['penn94', 'cornell5', 'amherst41']:
        return get_linkx_heter_dataset(root, name, device)
    else:
        raise NotImplementedError

def get_citation(root, dataset_name, device):
    """
    Load Citation Networks Datasets.
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    dataset_name = dataset_name.lower()

    for i in range(len(names)):
        with open(os.path.join(root, "citation/ind.{}.{}".format(dataset_name, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(root, "citation/ind.{}.test.index".format(dataset_name)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = preprocess_features(features)

    graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(graph)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    edge_index, _ = from_scipy_sparse_matrix(adj)
    edge_index = edge_index.to(device)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
  
    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float().to(device)
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1].to(device)

    nfeat = features.size(1)
    nclass = int(labels.max().item() + 1)

    train_mask_lst = []
    val_mask_lst = []
    test_mask_lst = []

    for i in range(10):
        splitstr = dataset_name + '_split_0.6_0.2_' + str(i) + '.npz'
        splits_file_path = os.path.join(root, 'citation', splitstr)
        with np.load(splits_file_path) as splits_file:
            train_mask_lst.append(splits_file['train_mask'])
            val_mask_lst.append(splits_file['val_mask'])
            test_mask_lst.append(splits_file['test_mask'])
    
    train_mask = torch.BoolTensor(train_mask_lst).transpose(1,0).to(device)
    val_mask = torch.BoolTensor(val_mask_lst).transpose(1,0).to(device)
    test_mask = torch.BoolTensor(test_mask_lst).transpose(1,0).to(device)

    return edge_index, features, labels, nfeat, nclass, train_mask, val_mask, test_mask

def get_heterophilic_dataset(root, dataset_name, device):
    dataset_name = dataset_name.lower()
    assert dataset_name in ['chameleon', 'squirrel']
    
    dataset = WikipediaNetwork(root, dataset_name, transform=T.NormalizeFeatures())
    
    data = dataset[0].to(device)
    nfeat = data.x.size(1)
    nclass = int(data.y.long().max().item() + 1)

    return data.edge_index, data.x, data.y.long(), nfeat, nclass, data.train_mask, data.val_mask, data.test_mask

class WikipediaNetwork(InMemoryDataset):
    r"""The Wikipedia networks used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to classify the nodes into five categories in term of the
    number of average monthly traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Chameleon"`,
            :obj:`"Squirrel"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f'

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel']

        super(WikipediaNetwork, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'] + [
            '{}_split_0.6_0.2_{}.npz'.format(self.name, i) for i in range(10)
        ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/{self.name}/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    train_mask = torch.zeros(label.size(0), dtype=torch.bool)
    valid_mask = torch.zeros(label.size(0), dtype=torch.bool)
    test_mask = torch.zeros(label.size(0), dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask[valid_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, valid_mask, test_mask


def get_linkx_heter_dataset(root, dataset, device):
    dataset = linkx_dataset.LINKXDataset(root, dataset)
    train_mask_list, val_mask_list, test_mask_list = [], [], []
    for i in range(10):
        train_mask, val_mask, test_mask = rand_train_test_idx(dataset.data.y)
        train_mask_list.append(train_mask), val_mask_list.append(val_mask), test_mask_list.append(test_mask) 
    dataset.data.train_mask, dataset.data.val_mask, dataset.data.test_mask = torch.stack(train_mask_list, dim=1), \
                    torch.stack(val_mask_list, dim=1), torch.stack(test_mask_list, dim=1)
    data = dataset.data.to(device)
    nfeat = data.x.size(1)
    nclass = int(data.y.long().max().item() + 1)

    return data.edge_index, data.x, data.y.long(), nfeat, nclass, data.train_mask, data.val_mask, data.test_mask