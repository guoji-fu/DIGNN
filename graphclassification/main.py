from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from sklearn.model_selection import StratifiedKFold

from models import *


def build_model(args, num_features, num_classes):
    if args.model == 'RW':
        model = DIGNN_RW(in_channels=num_features,
                            out_channels=num_classes,
                            hidden_channels=args.num_hid,
                            mu=args.mu,
                            max_iter=args.max_iter,
                            threshold=args.threshold,
                            num_layers=args.num_layers,
                            dropout=args.dropout)
    elif args.model == 'Neural':
        model = DIGNN_Neural(in_channels=num_features,
                                out_channels=num_classes,
                                hidden_channels=args.num_hid,
                                mu=args.mu,
                                max_iter=args.max_iter,
                                threshold=args.threshold,
                                num_layers=args.num_layers,
                                dropout=args.dropout)
    return model


def load_data(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.input)
    dataset = TUDataset(path, name=args.input).shuffle()
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = args.seed)
    idx_list = []
    for idx in skf.split(np.zeros(len(dataset.data.y)), dataset.data.y):
        idx_list.append(idx)

    if dataset.num_features ==0:
        num_features = 1
    else:
        num_features = dataset.num_features
    
    return dataset, idx_list, num_features, dataset.num_classes


def split_data(dataset, idx_list, fold_idx):
    train_idx, test_idx = idx_list[fold_idx]
    test_dataset = dataset[test_idx.tolist()]
    train_dataset = dataset[train_idx.tolist()]
    
    test_loader = DataLoader(test_dataset, batch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=128)

    return train_loader, test_loader


def train(model, optimizer, train_loader, device):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        edge_weight = torch.ones((data.edge_index.size(1), ), dtype=torch.float32, device=data.edge_index.device)

        adj_ori = torch.sparse.FloatTensor(data.edge_index, edge_weight, torch.Size([data.num_nodes, data.num_nodes])) #original adj
        if data.x is None:
            data.x = torch.sparse.sum(adj_ori, [0]).to_dense().unsqueeze(1).to(device)
        output = model(data.x, data.edge_index, edge_weight, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        edge_weight = torch.ones((data.edge_index.size(1), ), dtype=torch.float32, device=data.edge_index.device)

        adj_ori = torch.sparse.FloatTensor(data.edge_index, edge_weight, torch.Size([data.num_nodes, data.num_nodes])) #original adj

        if data.x is None:
            data.x = torch.sparse.sum(adj_ori, [0]).to_dense().unsqueeze(1).to(device)
        with torch.no_grad():
            output = model(data.x, data.edge_index, edge_weight, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def main(args):
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    if not os.path.exists(args.path):
        os.mkdir(args.path)

    result_name = f'model{args.input}_{args.epochs}_{args.num_hid}_l{args.num_layers}_lr{args.lr}_dp{args.dropout}_seed{args.seed}.txt'
    file_name = os.path.join(args.path, result_name)
    filep = open(file_name, 'w')
    filep.write(str(args) + '\n')

    dataset, idx_list, num_features, num_classes = load_data(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = [[] for i in range(args.runs)]
    for fold_idx in range(10):
        print(f'---------fold_idx: {fold_idx}-------------')
        filep.write(f'---------fold_idx: {fold_idx}-------------\n')
        # train_loader, test_loader = split_data(dataset, idx_list, fold_idx)
        model = build_model(args, num_features, num_classes)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        train_idx, test_idx = idx_list[fold_idx]
        test_dataset = dataset[test_idx.tolist()]
        train_dataset = dataset[train_idx.tolist()]

        test_loader = DataLoader(test_dataset, batch_size=128)
        train_loader = DataLoader(train_dataset, batch_size=128)

        for epoch in range(1, args.epochs+1):
            st_train = time.time()
            train_loss = train(model, optimizer, train_loader, device)
            train_time = time.time() - st_train
            train_acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)
            results[fold_idx].append(test_acc)
            print('Epoch: {:03d}, Train Loss: {:.7f}, '
                'Train Acc: {:.7f}, Test Acc: {:.7f}, Train_Time: {:.5f}'.format(epoch, train_loss,
                                                            train_acc, test_acc, train_time))
            filep.write('Epoch: {:03d}, Train Loss: {:.7f}, '
                        'Train Acc: {:.7f}, Test Acc: {:.7f}, Train_Time: {:.5f}\n'.format(epoch, train_loss,
                                                                    train_acc, test_acc, train_time))

    re_np = np.array(results)
    re_all = [re_np.max(1).mean(), re_np.max(1).std()]
    print('Graph classification averaged test accuracy and std for {} runs are: {}'.format(args.runs, re_all))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        default='MUTAG',     
                        choices=['MUTAG', 'PTC_MR', 'PROTEINS', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI'],
                        help='Input graph.')
    parser.add_argument('--model',
                        type=str,
                        default='Neural',
                        choices=['RW', 'Neural'],
                        help='GNN model')
    parser.add_argument('--runs',
                        type=int,
                        default=10,
                        help='Number of repeating experiments.')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', 
                        type=float, 
                        default=42,
                        help='Random seed.')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--num_hid', 
                        type=int, 
                        default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--path', 
                        type=str, 
                        default='./results_graph_cls/')
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=3,
                        help='Number of layers.')
    # Implicit setting
    parser.add_argument('--max_iter',
                        type=int,
                        default=50)
    parser.add_argument('--threshold',
                        type=float,
                        default=1e-6)

    # DirichletGNN
    parser.add_argument('--mu', 
                        type=float, 
                        default=4,
                        help='mu.')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(get_args())