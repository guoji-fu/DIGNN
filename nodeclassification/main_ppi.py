from os import X_OK
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score

import scipy.sparse as sp
import argparse
import time

from models_ppi import *


def build_model(args, num_features, num_classes):
    if args.model == 'RW':
        model = DIGNN_RW(in_channels=num_features,
                            out_channels=num_classes,
                            hidden_channels=args.num_hid,
                            mu=args.mu,
                            max_iter=args.max_iter,
                            threshold=args.threshold,
                            dropout=args.dropout,
                            num_layers=args.num_layers)
    elif args.model == 'Neural':
        model = DIGNN_Neural(in_channels=num_features,
                                out_channels=num_classes,
                                hidden_channels=args.num_hid,
                                mu=args.mu,
                                max_iter=args.max_iter,
                                threshold=args.threshold,
                                dropout=args.dropout)
    return model


def train(model, optimizer, train_loader, loss_op, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def test(model, loader, device):
    model.eval()
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

def main(args):
    # print('-----------plapgcn ppi %s %s %s-----------' % (args.mu, args.p, args.K))
    path = osp.join('dataset', 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    for run in range(args.runs):
        model = build_model(args, train_dataset.num_features, train_dataset.num_classes).to(device)
        loss_op = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        t1 = time.time()
        best_val_f1 = test_f1 = 0
        for epoch in range(1, args.epochs + 1):
            loss = train(model, optimizer, train_loader, loss_op, device)
            val_f1 = test(model, val_loader, device)
            tmp_test_f1 = test(model, test_loader, device)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                test_f1 = tmp_test_f1
            print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(epoch, loss, best_val_f1, test_f1))
        t2 = time.time()
        print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}, Time: {:.4f}'.format(epoch, loss, best_val_f1, test_f1, t2-t1))
        print('model:{} mu:{} max_iter:{}, Accuracy: {:.4f}, Time: {:.4f}'.format(args.model, args.mu, args.max_iter, test_f1, t2-t1))
        results.append(test_f1)
    print('{}: {:.4f}'.format(args.model, t2-t1))
    results = 100 * torch.Tensor(results)
    results = torch.Tensor(results)
    print(results)
    print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        default='cora',                    
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
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--num_hid', 
                        type=int, 
                        default=512,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.1,
                        help='Dropout rate (1 - keep probability).')
    
    # GAT
    parser.add_argument('--num_heads', 
                        type=int, 
                        default=8,
                        help='Number of heads.')
    
    # SGC & APPNP
    parser.add_argument('--K', 
                        type=int, 
                        default=4,
                        help='K.')
    
    # APPNP & GCNII
    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.5,
                        help='alpha.')
    
    # GCNII
    parser.add_argument('--theta',
                        type=float,
                        default=1.,
                        help='theta.')
    
    # GCNII & H2GCN
    parser.add_argument('--num_layers', 
                        type=int, 
                        default=4,
                        help='Number of layers.')
    
    # Implicit setting
    parser.add_argument('--max_iter',
                        type=int,
                        default=20,
                        help='maximum iteration numbers.')
    parser.add_argument('--threshold',
                        type=float,
                        default=1e-6,
                        help='threshold.')

    # DirichletGRAND
    parser.add_argument('--mu', 
                        type=float, 
                        default=2,
                        help='mu.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(get_args())