"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import torch
from random import randint


def create_dataset():
    Gs = list()
    y = list()

    ############## Task 5
    Gs += [nx.fast_gnp_random_graph(randint(10, 20), 0.2) for _ in range(50)] + [
        nx.fast_gnp_random_graph(randint(10, 20), 0.4) for _ in range(50)
    ]
    y += [0] * 50 + [1] * 50

    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
