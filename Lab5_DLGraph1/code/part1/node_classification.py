"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist(
    "../data/karate.edgelist", delimiter=" ", nodetype=int, create_using=nx.Graph()
)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt("../data/karate_labels.txt", delimiter=",", dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i, 0]] = class_labels[i, 1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network
import matplotlib.pyplot as plt

color_map = ["red" if label == 1 else "blue" for label in y]

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw_networkx(
    G,
    pos,
    node_color=color_map,
    with_labels=True,
    node_size=600,
    font_size=10,
    font_color="white",
)
plt.title("Visualization of the Karate Network")
plt.show()


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i, :] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[: int(0.8 * n)]
idx_test = idx[int(0.8 * n) :]

X_train = embeddings[idx_train, :]
X_test = embeddings[idx_test, :]

y_train = y[idx_train]
y_test = y[idx_test]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# ############## Task 7
# # Trains a logistic regression classifier and use it to make predictions

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Measure accuracy
print("Accuracy with logistic regression:", accuracy_score(y_test, y_pred))


# ############## Task 8
# # Generates spectral embeddings
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
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


clustering = spectral_clustering(G, 2)[idx_test]
print("Accuracy with spectral clustering:", accuracy_score(y_test, clustering))
