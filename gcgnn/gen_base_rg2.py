import os
import pickle
import argparse
import numpy as np
import networkx as nx
    
def graph2rg2mean(G):
    """
    Calculate baseline Rg2 from a networkx graph based on
    Rg2 = trace(L+)/n
    L+ is the pseudo-inverse of the graph Lapaliacan and n is the number of nodes.
    
    Args:
    - G: NetworkX graph.

    Returns:
    - rg2: Baseline squared raidus of gyration.
    - pattern_types: Type of patterns in the graph (0, 1, or 2) representing:
        0: If only type 0 nodes are present.
        1: If only type 1 nodes are present.
        2: If both type 0 and type 1 nodes are present.
    """
    
    L             =  nx.laplacian_matrix(G).toarray()
    L_plus        =  np.linalg.pinv(L)
    trace_L_plus  =  np.trace(L_plus)
    dp            =  G.number_of_nodes() # number of nodes/degree of polymerization
    rg2           =  trace_L_plus / dp
    pattern_types =  np.unique([G.nodes[node]['type'] for node in G.nodes])
    
    if 0 in pattern_types and 1 in pattern_types:
        pattern_types = 2
    elif 0 in pattern_types and 1 not in pattern_types:
        pattern_types = 0
    elif 0 not in pattern_types and 1 in pattern_types:
        pattern_types = 1

    return rg2, pattern_types


def graph2rg2var(G):
    """
    Calculate baseline Rg2 from a networkx graph based on
    Rg2 = trace(L+)/n
    L+ is the pseudo-inverse of the graph Lapaliacan and n is the number of nodes.
    
    Args:
    - G: NetworkX graph.

    Returns:
    - rg2: Baseline squared raidus of gyration.
    - pattern_types: Type of patterns in the graph (0, 1, or 2) representing:
        0: If only type 0 nodes are present.
        1: If only type 1 nodes are present.
        2: If both type 0 and type 1 nodes are present.
    """
    
    L             =  nx.laplacian_matrix(G).toarray()
    L_plus        =  np.linalg.pinv(L)
    L_plus_2      =  L_plus ** 2
    trace_L_plus  =  np.trace(L_plus_2)
    dp            =  G.number_of_nodes() # number of nodes/degree of polymerization
    rg2           =  trace_L_plus * (2 / 3 / dp **2) 
    pattern_types =  np.unique([G.nodes[node]['type'] for node in G.nodes])
    
    if 0 in pattern_types and 1 in pattern_types:
        pattern_types = 2
    elif 0 in pattern_types and 1 not in pattern_types:
        pattern_types = 0
    elif 0 not in pattern_types and 1 in pattern_types:
        pattern_types = 1

    return rg2, pattern_types


def main(args):
    start = args.start
    end   = start + 20
    
    graph_data  = {}

    input_file = os.path.join(args.data_dir, f"pattern_graph_data_{start}_{end}_rg.pickle")
    output_file  = os.path.join(args.data_dir, f"rg2_baseline_{start}.pickle")
        
    with open(input_file, "rb") as handle:
        graph_data[start] = pickle.load(handle)

    rg2_mean = np.array([graph2rg2mean(G) for G in graph_data[start]])
    rg2_var = np.array([graph2rg2var(G) for G in graph_data[start]])
    
    with open(output_file, "wb") as handle:
        pickle.dump(rg2_mean, handle)
        pickle.dump(rg2_var, handle)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for gen_base_rg2")
    parser.add_argument('--start', type=int, default=40, help="Bead value")
    parser.add_argument('--data_dir', type=str, default='data', help="Data directory")
    args = parser.parse_args()
    main(args)
