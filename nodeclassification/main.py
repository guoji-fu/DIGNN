import torch
import argparse
import time

from data import get_data
from models import *
from baselines import *


def build_model(args, edge_index, num_nodes, num_features, num_classes, device):
    if args.model == 'RW':
        model = DIGNN_RW(in_channels=num_features,
                                    out_channels=num_classes,
                                    hidden_channels=args.num_hid,
                                    edge_index=edge_index,
                                    num_nodes=num_nodes,
                                    device=device,
                                    mu=args.mu,
                                    max_iter=args.max_iter,
                                    threshold=args.threshold,
                                    dropout=args.dropout,
                                    preprocess=args.preprocess)
    elif args.model == 'Neural':
        model = DIGNN_Neural(in_channels=num_features,
                                    out_channels=num_classes,
                                    hidden_channels=args.num_hid,
                                    edge_index=edge_index,
                                    num_nodes=num_nodes,
                                    device=device,
                                    mu=args.mu,
                                    max_iter=args.max_iter,
                                    threshold=args.threshold,
                                    dropout=args.dropout,
                                    preprocess=args.preprocess)
    elif args.model == 'mlp':
        model = MLPNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'gcn':
        model = GCNNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'sgc':
        model = SGCNet(in_channels=num_features,
                        out_channels=num_classes,
                        K=args.K)
    elif args.model == 'gat':
        model = GATNet(in_channels=num_features,
                       out_channels=num_classes,
                       num_hid=args.num_hid,
                       num_heads=args.num_heads,
                       dropout=args.dropout)
    elif args.model == 'jk':
        model = JKNet(in_channels=num_features,
                      out_channels=num_classes,
                      num_hid=args.num_hid,
                      K=args.K,
                      alpha=args.alpha,
                      dropout=args.dropout)
    elif args.model == 'appnp':
        model = APPNPNet(in_channels=num_features,
                         out_channels=num_classes,
                         num_hid=args.num_hid,
                         K=args.K,
                         alpha=args.alpha,
                         dropout=args.dropout)
    elif args.model == 'gcnii':
        model = GCNIINet(in_channels=num_features,
                         out_channels=num_classes,
                         hidden_channels=args.num_hid,
                         num_layers=args.num_layers,
                         dropout=args.dropout,
                         alpha=args.alpha,
                         theta=args.theta)
    elif args.model == 'h2gcn':
        model = H2GCNNet(in_channels=num_features,
                         out_channels=num_classes,
                         hidden_channels=args.num_hid,
                         edge_index=edge_index,
                         num_layers=args.num_layers,
                         dropout=args.dropout)

    return model

def train(model, optimizer, x, y, edge_index, train_mask):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(x, edge_index)[train_mask], y[train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(model, x, y, edge_index, train_mask, val_mask, test_mask):
    model.eval()
    logits, accs = model(x, edge_index), []
    train_pred = logits[train_mask].max(1)[1]
    train_acc = train_pred.eq(y[train_mask]).sum().item() / train_mask.sum().item()
    accs.append(train_acc)
    val_pred = logits[val_mask].max(1)[1]
    val_acc = val_pred.eq(y[val_mask]).sum().item() / val_mask.sum().item()
    accs.append(val_acc)
    test_pred = logits[test_mask].max(1)[1]
    test_acc = test_pred.eq(y[test_mask]).sum().item() / test_mask.sum().item()
    accs.append(test_acc)
    return accs

def main(args):
    # print(args)
    print(args.input, args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    edge_index, features, labels, in_channels, out_channels, \
    train_mask, val_mask, test_mask = get_data('data', args.input, device)
    num_nodes = features.size(0)
    results = []
    for run in range(args.runs):
        idx_train, idx_val, idx_test = train_mask[:, run], val_mask[:, run], test_mask[:, run]
        model = build_model(args, edge_index, num_nodes, in_channels, out_channels, device)
        model = model.to(device)
        # data = data.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        
        t1 = time.time()
        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs+1):
            # t01 = time.time()
            train(model, optimizer, features, labels, edge_index, idx_train)
            # t02 = time.time()
            train_acc, val_acc, tmp_test_acc = test(model, features, labels, edge_index, idx_train, idx_val, idx_test)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
            # print(t02 - t01)
        t2 = time.time()
        print('{}, {}, Accuacy: {:.4f}, Time: {:.4f}'.format(args.model, args.input, test_acc, t2-t1))
        results.append(test_acc)
    results = 100 * torch.Tensor(results)
    print(results)
    print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        default='chameleon',   
                        choices=['chameleon', 'squirrel', 'penn94', 'cornell5', 'amherst41', 'cora', 'citeseer', 'pubmed'],
                        help='Input graph.')
    parser.add_argument('--train_rate', 
                        type=float, 
                        default=0.05,
                        help='Training rate.')
    parser.add_argument('--val_rate', 
                        type=float, 
                        default=0.05,
                        help='Validation rate.')
    parser.add_argument('--model',
                        type=str,
                        default='Neural',
                        choices=['mlp', 'gcn', 'gat', 'jk', 'appnp', 'gcnii', 'h2gcn', 'gind', 'RW', 'Neural'],
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
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--num_hid', 
                        type=int, 
                        default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.5,
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
                        default=1,
                        help='Number of layers.')
    
    # Implicit setting
    parser.add_argument('--max_iter',
                        type=int,
                        default=20)
    parser.add_argument('--threshold',
                        type=float,
                        default=1e-6)

    # DirichletGRAND
    parser.add_argument('--mu', 
                        type=float, 
                        default=2.2,
                        help='mu.')
    # parser.add_argument('--learn_adj',
    #                     action='store_true',
    #                     default=False)
    parser.add_argument('--preprocess',
                        type=str,
                        default='adj')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(get_args())