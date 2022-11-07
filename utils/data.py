import networkx as nx
import numpy as np
import scipy
import pickle
import torch
import numpy as np


def load_data(dataset):
    assert dataset in ['DBLP', 'ACM', 'IMDB'], "Invalid dataset."
    if dataset == 'DBLP':
        return load_DBLP_data()
    elif dataset == 'ACM':
        return load_ACM_data()
    elif dataset == 'IMDB':
        return load_IMDB_data()


def load_IMDB_data(prefix='data/preprocessed/IMDB_processed'):
    # in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    # adjlist00 = [line.strip() for line in in_file]
    # adjlist00 = adjlist00[3:]
    # in_file.close()
    # in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    # adjlist01 = [line.strip() for line in in_file]
    # adjlist01 = adjlist01[3:]
    # in_file.close()
    # num_nodes = len(adjlist00)
    # adj_010 = np.zeros([num_nodes, num_nodes])
    # adj_020 = np.zeros([num_nodes, num_nodes])
    # for row in adjlist00:
    #     adj = list(map(int, row.split(' ')))
    #     node_idx = adj[0]
    #     for i in range(1, len(adj)):
    #         adj_010[node_idx, adj[i]] = 1
    # scipy.sparse.save_npz(prefix + '/adj_010.npz', scipy.sparse.csr_matrix(adj_010))
    # for row in adjlist01:
    #     adj = list(map(int, row.split(' ')))
    #     node_idx = adj[0]
    #     for i in range(1, len(adj)):
    #         adj_020[node_idx, adj[i]] = 1
    # scipy.sparse.save_npz(prefix + '/adj_020.npz', scipy.sparse.csr_matrix(adj_020))

    adj_010 = scipy.sparse.load_npz(prefix + '/adj_010.npz')
    adj_020 = scipy.sparse.load_npz(prefix + '/adj_020.npz')

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz')
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    adjMM = scipy.sparse.load_npz(prefix + '/adjMM.npz')
    adjMM_wsl = scipy.sparse.load_npz(prefix + '/adjMM_wsl.npz')
    adjMM_wsl_2 = scipy.sparse.load_npz(prefix + '/adjMM_wsl_2.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adj_010, adj_020], \
           [mat2tensor(features_0).to_dense(), mat2tensor(features_1).to_dense(), mat2tensor(features_2).to_dense()],\
           adjM, \
           adjMM, \
           adjMM_wsl, \
           adjMM_wsl_2, \
           type_mask,\
           labels,\
           train_val_test_idx

def load_ACM_data(prefix='data/preprocessed/ACM_processed'):
    # in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    # adjlist00 = [line.strip() for line in in_file]
    # adjlist00 = adjlist00[3:]
    # in_file.close()
    # in_file = open(prefix + '/0/0-2-0.adjlist', 'r')
    # adjlist01 = [line.strip() for line in in_file]
    # adjlist01 = adjlist01[3:]
    # in_file.close()
    # num_nodes = len(adjlist00)
    # adj_010 = np.zeros([num_nodes, num_nodes])
    # adj_020 = np.zeros([num_nodes, num_nodes])
    # for row in adjlist00:
    #     adj = list(map(int, row.split(' ')))
    #     node_idx = adj[0]
    #     for i in range(1, len(adj)):
    #         adj_010[node_idx, adj[i]] = 1
    # scipy.sparse.save_npz(prefix + '/adj_010.npz', scipy.sparse.csr_matrix(adj_010))
    # for row in adjlist01:
    #     adj = list(map(int, row.split(' ')))
    #     node_idx = adj[0]
    #     for i in range(1, len(adj)):
    #         adj_020[node_idx, adj[i]] = 1
    # scipy.sparse.save_npz(prefix + '/adj_020.npz', scipy.sparse.csr_matrix(adj_020))
    # print(scipy.sparse.csr_matrix(adj_010).shape)
    # print(scipy.sparse.csr_matrix(adj_010).getnnz())
    # print(scipy.sparse.csr_matrix(adj_010))

    adj_010 = scipy.sparse.load_npz(prefix + '/adj_010.npz')
    adj_020 = scipy.sparse.load_npz(prefix + '/adj_020.npz')

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    adjMM = scipy.sparse.load_npz(prefix + '/adjMM_rgcn.npz')
    adjMM_wsl = scipy.sparse.load_npz(prefix + '/adjMM_wsl.npz')
    adjMM_wsl_2 = scipy.sparse.load_npz(prefix + '/adjMM_wsl_2.npz')

    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adj_010, adj_020], \
           [features_0, features_1, features_2], \
           adjM, \
           adjMM, \
           adjMM_wsl, \
           adjMM_wsl_2, \
           type_mask, \
           labels, \
           train_val_test_idx


def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
    # in_file = open(prefix + '/0/0-1-0.adjlist', 'r')
    # adjlist00 = [line.strip() for line in in_file]
    # adjlist00 = adjlist00[3:]
    # in_file.close()
    # in_file = open(prefix + '/0/0-1-2-1-0.adjlist', 'r')
    # adjlist01 = [line.strip() for line in in_file]
    # adjlist01 = adjlist01[3:]
    # in_file.close()
    # in_file = open(prefix + '/0/0-1-3-1-0.adjlist', 'r')
    # adjlist02 = [line.strip() for line in in_file]
    # adjlist02 = adjlist02[3:]
    # in_file.close()

    # num_nodes = len(adjlist00)
    # adj_010 = np.zeros([num_nodes, num_nodes])
    # adj_01210 = np.zeros([num_nodes, num_nodes])
    # adj_01310 = np.zeros([num_nodes, num_nodes])
    # for row in adjlist00:
    #     adj = list(map(int, row.split(' ')))
    #     node_idx = adj[0]
    #     for i in range(1, len(adj)):
    #         adj_010[node_idx, adj[i]] = 1
    # scipy.sparse.save_npz(prefix + '/adj_010.npz', scipy.sparse.csr_matrix(adj_010))
    # for row in adjlist01:
    #     adj = list(map(int, row.split(' ')))
    #     node_idx = adj[0]
    #     for i in range(1, len(adj)):
    #         adj_01210[node_idx, adj[i]] = 1
    # scipy.sparse.save_npz(prefix + '/adj_01210.npz', scipy.sparse.csr_matrix(adj_01210))
    # for row in adjlist02:
    #     adj = list(map(int, row.split(' ')))
    #     node_idx = adj[0]
    #     for i in range(1, len(adj)):
    #         adj_01310[node_idx, adj[i]] = 1
    # scipy.sparse.save_npz(prefix + '/adj_01310.npz', scipy.sparse.csr_matrix(adj_01310))
    
    adj_010 = scipy.sparse.load_npz(prefix + '/adj_010.npz')
    adj_01210 = scipy.sparse.load_npz(prefix + '/adj_01210.npz')
    adj_01310 = scipy.sparse.load_npz(prefix + '/adj_01310.npz')

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = np.load(prefix + '/features_2.npy')
    features_3 = np.eye(20, dtype=np.float32)

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    adjMM = scipy.sparse.load_npz(prefix + '/adjMM.npz')
    adjMM_wsl = scipy.sparse.load_npz(prefix + '/adjMM_wsl.npz')
    adjMM_wsl_2 = scipy.sparse.load_npz(prefix + '/adjMM_wsl_2.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    return [adj_010, adj_01210, adj_01310], \
           [features_0, features_1, features_2, features_3], \
           adjM, \
           adjMM, \
           adjMM_wsl, \
           adjMM_wsl_2, \
           type_mask,\
           labels,\
           train_val_test_idx


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)

