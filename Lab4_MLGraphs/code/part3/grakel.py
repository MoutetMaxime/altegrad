import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings

warnings.filterwarnings("ignore")


def load_file(filename):
    labels = []
    docs = []

    with open(filename, encoding="utf8", errors="ignore") as f:
        for line in f:
            content = line.split(":")
            labels.append(content[0])
            docs.append(content[1][:-1])

    return docs, labels


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs):
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])

    return preprocessed_docs


def get_vocab(train_docs, test_docs):
    vocab = dict()

    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


path_to_train_set = "train_5500_coarse.label"
path_to_test_set = "TREC_10_coarse.label"

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


import networkx as nx
import matplotlib.pyplot as plt


# Task 11
def create_graphs_of_words(docs, vocab, window_size, id2w):
    graphs = list()
    for idx, doc in enumerate(docs):
        G = nx.Graph()
        for i in range(len(doc)):
            G.add_node(vocab[doc[i]], label=id2w[vocab[doc[i]]])

        for i in range(len(doc)):
            for j in range(max(0, i - window_size), min(len(doc), i + window_size + 1)):
                u, v = vocab[doc[i]], vocab[doc[j]]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)
        graphs.append(G)

    return graphs


# Create graph-of-words representations
id2w = {v: k for k, v in vocab.items()}
G_train_nx = create_graphs_of_words(train_data, vocab, 3, id2w)
G_test_nx = create_graphs_of_words(test_data, vocab, 3, id2w)

example_graph = G_train_nx[3]
labels = nx.get_node_attributes(example_graph, "label")  # Extract labels from the graph

# Plot with string labels
print("Example of graph-of-words representation of document")
nx.draw_networkx(example_graph, labels=labels, with_labels=True)
plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Task 12

# Transform networkx graphs to grakel representations
G_train = graph_from_networkx(G_train_nx, node_labels_tag="label")
G_test = graph_from_networkx(G_test_nx, node_labels_tag="label")

# Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_iter=3, base_graph_kernel=VertexHistogram)

# Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

# Task 13

# Train an SVM classifier and make predictions
svm = SVC(kernel="precomputed")  # Use precomputed kernel
svm.fit(K_train, y_train)  # y_train should be your training labels

# Predict using the test kernel matrix
y_pred = svm.predict(K_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)  # y_test should be your test labels
print(f"Test Accuracy with WeisfeilerLehman Kernel: {accuracy:.2f}")  # 0.95


# Task 14
from grakel.kernels import ShortestPath

G_train = graph_from_networkx(G_train_nx, node_labels_tag="label")
G_test = graph_from_networkx(G_test_nx, node_labels_tag="label")

gk_sp = ShortestPath()
K_train_sp = gk_sp.fit_transform(G_train)
K_test_sp = gk_sp.transform(G_test)

svm.fit(K_train_sp, y_train)
y_pred_sp = svm.predict(K_test_sp)
accuracy_sp = accuracy_score(y_test, y_pred_sp)
print(f"Test Accuracy with Shortest Path Kernel: {accuracy_sp:.2f}")  # 0.97


from grakel.kernels import GraphletSampling

G_train = graph_from_networkx(G_train_nx, node_labels_tag="label")
G_test = graph_from_networkx(G_test_nx, node_labels_tag="label")

gk_graphlet = GraphletSampling()
K_train_graphlet = gk_graphlet.fit_transform(G_train)
K_test_graphlet = gk_graphlet.transform(G_test)

svm.fit(K_train_graphlet, y_train)
y_pred_graphlet = svm.predict(K_test_graphlet)
accuracy_graphlet = accuracy_score(y_test, y_pred_graphlet)
print(f"Test Accuracy with Graphlet Kernel: {accuracy_graphlet:.2f}")
