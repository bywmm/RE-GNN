import time
import random
import argparse

import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from model import REGAT, REGCN

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from utils.tools import evaluate_results_nc


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    acc = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return acc, micro_f1, macro_f1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        print('Using CUDA')
        torch.cuda.manual_seed(seed)

set_seed(123)

def run(args):
    features_list, adjM, adjMM, adjMM_wsl, adjMM_wsl_2, type_mask, labels, train_val_test_idx = load_data(args.dataset)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]

    # 0 - loaded features
    if args.feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    # 1 - only target node features (zero vec for others)
    elif args.feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    # 2 - only target node features (id vec for others)
    elif args.feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    # 3 - all id vec
    elif args.feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    print(labels.size())
    print(features_list[0].size())
    print(adjMM.shape)

    g = dgl.DGLGraph(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    print(type(g))
    N = g.number_of_nodes()

    num_etype = adjMM.max()
    num_ntype = type_mask.max() + 1

    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(adjMM_wsl_2[(u, v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []
    heads = ([args.num_heads] * args.num_layers) + [1]
    for _ in range(args.repeat):
        num_classes = labels.max().item()+1
        if args.model == 'regat':
            net = REGAT(g, num_etype+num_ntype, args.num_layers, args.hidden_dim, args.hidden_dim,
                         num_classes, heads, F.elu, args.dropout, args.dropout, 0.01, False, in_dims)
        elif args.model == 'regcn':
            net = REGCN(g, num_etype+num_ntype, args.hidden_dim, args.hidden_dim, num_classes,
                         args.num_layers, F.elu, args.dropout, in_dims)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=False,
                                       save_path='checkpoint/checkpoint_{}.pt'.format(args.save_postfix))
        for epoch in range(args.epochs):
            t_start = time.time()
            # training
            net.train()
            # forward
            logits = net(features_list, e_feat)
            train_loss = loss_fn(logits[train_idx], labels[train_idx])

            # autograd
            optimizer.zero_grad()
            train_loss.backward()

            optimizer.step()

            net.eval()
            with torch.no_grad():
                logits = net(features_list, e_feat)
                val_loss = loss_fn(logits[val_idx], labels[val_idx])
                val_acc, val_mif1, val_maf1 = score(logits[val_idx], labels[val_idx])
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Train_Loss {:.4f} | Val_Loss {:.4f}, Val_mif1 {:.4f}, Val_maf1 {:.4f} | Time(s) {:.4f}'.format(
                epoch, train_loss.item(), val_loss.item(), val_mif1, val_maf1, t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(args.save_postfix)))
        net.eval()
        #test_embeddings = []
        with torch.no_grad():
            logits = net(features_list, e_feat)
            test_embeddings = logits[test_idx]
            print('-----------')
            print(score(logits[test_idx],labels[test_idx]))
            print('-----------')
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
                test_embeddings.cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=num_classes)
        svm_macro_f1_lists.append(svm_macro_f1_list)
        svm_micro_f1_lists.append(svm_micro_f1_list)
        nmi_mean_list.append(nmi_mean)
        nmi_std_list.append(nmi_std)
        ari_mean_list.append(ari_mean)
        ari_std_list.append(ari_std)

    # print out a summary of the evaluations
    svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
    svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))

    print('----------------------------------------------------------------')
    print('SVM tests summary')
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the ACM dataset')
    ap.add_argument('--dataset', default='ACM', help='ACM, DBLP, IMDB')
    ap.add_argument('--model', default='regcn', help='regcn, regat')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--num_layers', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=200, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=50, help='Patience. Default is 5.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--save-postfix', default='ACM', help='Postfix for the saved model and result. Default is DBLP.')
    ap.add_argument('--device', type=int, default=5)
    ap.add_argument('--dropout', type=float, default=0.6)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight_decay', type=float, default=0.001)


    args = ap.parse_args()
    print(args)

    run(args)
