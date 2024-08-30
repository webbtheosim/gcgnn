import networkx as nx
import numpy as np
from collections import defaultdict

def relabel(G):
    mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    return G

def get_adj(G):
    adj = nx.to_numpy_array(G)
    return adj

def get_desc(G):
    x1 = nx.number_of_nodes(G)
    x2 = nx.number_of_edges(G)
    x3 = nx.algebraic_connectivity(G)
    x4 = nx.diameter(G)
    x5 = nx.radius(G)
    degrees = [degree for _, degree in G.degree()]
    x6  = sum(degrees) / len(G.nodes())
    x7  = np.mean(list(nx.average_neighbor_degree(G).values()))
    x8  = nx.density(G)
    x9  = np.mean(list(nx.degree_centrality(G).values()))
    x10 = np.mean(list(nx.betweenness_centrality(G).values()))
    x11 = nx.degree_assortativity_coefficient(G)
    return np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11])


def is_isomorphic(G_list):
    dup_list = []
    print(f"Isomorphic checking started ...")
    for i in range(len(G_list)):
        for j in range(i + 1, len(G_list)):
            if nx.is_isomorphic(G_list[i], G_list[j]):
                print(f"Graph {i} and {j} are isomorphic")
                dup_list.append([i, j])
    print(f"Isomorphic checking finished ...")
    
    return np.array(dup_list)

def select_node_count(graph, label, graph_desc, poly_label, max_node_count=5):
    selected_graphs = []
    selected_labels = []
    selected_graph_descs = []
    selected_poly_labels = []

    node_count_dict = defaultdict(int)

    for i, G in enumerate(graph):
        node_count = G.number_of_nodes()

        if node_count_dict[node_count] < max_node_count:
            node_count_dict[node_count] += 1
            selected_graphs.append(G)
            selected_labels.append(label[i])
            selected_graph_descs.append(graph_desc[i])
            selected_poly_labels.append(poly_label[i])
        
    return (selected_graphs, selected_labels, selected_graph_descs, selected_poly_labels)