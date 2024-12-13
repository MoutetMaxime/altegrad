"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
G = nx.read_edgelist("../datasets/CA-HepTh.txt", delimiter="\t")

print("=== Description of G ===")
print("Nodes: ", len(G.nodes))
print("Edges: ", len(G.edges))
print()
############## Task 2

print("=== Connected components ===")
sorted_connected_components = [
    len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)
]

print("Number: ", len(sorted_connected_components))
print(
    "Size of the largest: ",
    sorted_connected_components[0],
    f"({100 * sorted_connected_components[0] / len(G.nodes):.1f}% of total nodes)",
)
