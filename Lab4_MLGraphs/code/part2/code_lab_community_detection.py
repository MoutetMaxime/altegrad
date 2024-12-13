"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    n = len(G.nodes)
    A = nx.adjacency_matrix(G)

    # Degree matrix
    degrees = [d[1] if d[1] > 0 else 1 for d in G.degree()]
    inv_D = diags([1 / d for d in degrees])

    # Laplacian matrix
    L = eye(n) - inv_D @ A

    w, v = eigs(L, k=k, which="SM")

    # Take real part to handle potential complex values due to numerical errors
    U = np.real(v)
    kmeans = KMeans(n_clusters=k)
    clustering = kmeans.fit_predict(U)
    return clustering


############## Task 4
G = nx.read_edgelist("../datasets/CA-HepTh.txt", delimiter="\t")
largest_cc = max(nx.connected_components(G), key=len)
subG = G.subgraph(largest_cc).copy()
clustering = spectral_clustering(subG, 50)


############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    k = len(clustering)
    m = len(G.edges)

    res = 0
    for c in range(k):
        cluster = [node for node, label in zip(G.nodes(), clustering) if label == c]
        lc = G.subgraph(cluster).number_of_edges()
        dc = sum(dict(G.degree(cluster)).values())

        res += lc / m - (dc / (2 * m)) ** 2

    return res


############## Task 6
print(modularity(subG, clustering))  # 0.196
random_clustering = [randint(0, 50) for _ in range(len(subG.nodes))]
print(modularity(subG, random_clustering))  # 0.0002
