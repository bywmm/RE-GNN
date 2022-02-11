import networkx as nx
import numpy as np
import scipy

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

    return [mat2tensor(features_0).to_dense(), mat2tensor(features_1).to_dense(), mat2tensor(features_2).to_dense()],\
           adjM, \
           adjMM, \
           adjMM_wsl, \
           adjMM_wsl_2, \
           type_mask,\
           labels,\
           train_val_test_idx

def load_ACM_data(prefix='data/preprocessed/ACM_processed'):
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

    return [features_0, features_1, features_2], \
           adjM, \
           adjMM, \
           adjMM_wsl, \
           adjMM_wsl_2, \
           type_mask, \
           labels, \
           train_val_test_idx


def load_DBLP_data(prefix='data/preprocessed/DBLP_processed'):
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

    return [features_0, features_1, features_2, features_3],\
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

