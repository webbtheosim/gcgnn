import os
import pickle
import torch

import numpy as np
import networkx as nx

from tqdm import tqdm
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

from gcgnn.model_data import str2num, preprocess_topo
from gcgnn.model_data import load_graph_rg_data, shuffle_data

SEED = 42

def create_data(args, graphs, beads, rg2_means, rg2_vars):
    """ Create PyTorch Geometric data objects from the input data."""
    x_graphs     = []
    rg2_m_bases  = []
    rg2_s_bases  = []
    y_rg2_means  = []
    y_rg2_vars   = []

    for graph, bead, rg2_mean, rg2_var in zip(graphs, beads, rg2_means, rg2_vars):
        
        x_graphs.append(graph)
        y_rg2_means.append(rg2_mean)
        y_rg2_vars.append(rg2_var)
        with open(os.path.join(args.DATA_DIR, f"rg2_baseline_{bead}_new.pickle"), "rb") as handle:
            m_base = pickle.load(handle)[:, 0]
            s_base = pickle.load(handle)[:, 0]
            rg2_m_bases.append(m_base)
            rg2_s_bases.append(s_base)


    output = [[] for _ in range(len(graphs))]

    for idx, x_graph in enumerate(x_graphs):
        for i, G in tqdm(enumerate(x_graph), total=len(x_graph)):
            n_node = G.number_of_nodes()
            node_types = [str2num(G.nodes[node]['type']) for node in G.nodes]
            nt = np.array(node_types)
            nt = nt[..., None]
            
            node_degrees = [G.degree(i) for i in range(n_node)]
            nd = np.array(node_degrees)
            nd = nd[..., None]
            
            average_neighbor_degree = nx.average_neighbor_degree(G)
            avd = np.array([average_neighbor_degree[i] for i in range(n_node)])
            avd = avd[..., None]

            node_features = np.concatenate((nt, nd, avd), axis=1)
            
            patt_types = np.unique(node_types)
            
            if len(patt_types) == 2:                            # mixture
                patt_types = 2
            elif len(patt_types) == 1 and patt_types[0] == 0.16: # pure
                patt_types = 0
            elif len(patt_types) == 1 and patt_types[0] == 1.0:
                patt_types = 1
            else:
                raise ValueError("Invalid pattern type value")
            
            x       = torch.tensor(node_features, dtype=torch.float)
            y_mean  = torch.tensor([[y_rg2_means[idx][i]]], dtype=torch.float)
            y_std   = torch.tensor([[y_rg2_vars[idx][i]]], dtype=torch.float)
            m_base  = torch.tensor([[rg2_m_bases[idx][i]]], dtype=torch.float)
            s_base  = torch.tensor([[rg2_s_bases[idx][i]]], dtype=torch.float)
            
            edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
            
            pattern    = torch.tensor(patt_types, dtype=torch.float)
            
            data       = Data(x=x, 
                              edge_index=edge_index, 
                              y_mean=y_mean,
                              y_std=y_std,
                              m_base=m_base,
                              s_base=s_base,
                              pattern=pattern)
            
            output[idx].append(data)

    return output


def load_preprocess_data(args):
    """ Load and preprocess the input data."""
    graphs    = []
    topos     = []
    rg2_means = []
    rg2_vars  = []

    beads  = [40, 90, 190]

    for bead in beads:
        file_dir = os.path.join(args.DATA_DIR, f"pattern_graph_data_{bead}_{bead+20}_rg_new.pickle")

        graph, topo, rg2_mean, rg2_var = load_graph_rg_data(file_dir)

        rg2_mean  = rg2_mean
        rg2_var   = rg2_var ** 0.5

        topo = preprocess_topo(topo)

        graphs.append(graph)
        topos.append(topo)
        rg2_means.append(rg2_mean)
        rg2_vars.append(rg2_var)

    data = create_data(args, graphs, beads, rg2_means, rg2_vars)

    shuffle_data(data, topos)

    return data, topos


def load_networkx_data(DATA_DIR="/scratch/gpfs/sj0161/delta_pattern/"):
    """ Load the networkx data."""
    data_ranges = [(40, 60), (90, 110), (190, 210)]

    graph_data = {}
    label_data = {}
    desc_data  = {}
    meta_data  = {}
    mode_data  = {}
    rg2_mean   = {}
    rg2_std    = {}

    for start, end in data_ranges:
        filename = os.path.join(DATA_DIR, f"pattern_graph_data_{start}_{end}_rg_new.pickle")

        with open(filename, "rb") as handle:
            graph_data[start] = pickle.load(handle)
            label_data[start] = pickle.load(handle)
            desc_data[start]  = pickle.load(handle)
            meta_data[start]  = pickle.load(handle)
            mode_data[start]  = pickle.load(handle)
            rg2_mean[start]   = pickle.load(handle)
            rg2_std[start]    = pickle.load(handle) ** 0.5

        label_data[start] = np.array(label_data[start])
        label_data[start][label_data[start] == 'stara'] = 'star'
        label_data[start][label_data[start] == 'bottlebrush'] = 'comb' 

    return graph_data, label_data, desc_data, meta_data, mode_data, rg2_mean, rg2_std


def load_data(args):
    """ Load the input data for NN training and testing """
    DATA_FILE  = os.path.join(args.DATA_DIR, "delta_data_v0314.pickle")
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as handle:
            data = pickle.load(handle)
            topo = pickle.load(handle)
    else:
        data, topo = load_preprocess_data(args)
        with open(DATA_FILE, "wb") as handle:
            pickle.dump(data, handle)
            pickle.dump(topo, handle)
    
    data_40, data_90, data_190 = data
    topo_40, topo_90, topo_190 = topo
    
    args.input_dim = int(data_40[0].x.shape[-1])
    
    new_data_sets = [[], [], []]

    for idx, dataset in enumerate(data):
        new_dataset = new_data_sets[idx]
        for d in dataset:
            if args.if_log == 1:
                d.y = torch.log10(d.y_mean) if args.y_type == "mean" else torch.log10(d.y_std)
                d.base = torch.log10(d.m_base) if args.y_type == "mean" else torch.log10(d.s_base)
            else:
                d.y = d.y_mean if args.y_type == "mean" else d.y_std
                d.base = d.m_base if args.y_type == "mean" else d.s_base
            new_dataset.append(d)
            
    data_40, data_90, data_190 = new_data_sets
    
    if args.split_type == 0:
        data_train_val = data_40 + data_90
        topo_train_val = topo_40 + topo_90
        data_test  = data_190
        topo_test  = topo_190
        
    elif args.split_type == 1:
        data_train_val = data_90 + data_190
        topo_train_val = topo_90 + topo_190
        data_test  = data_40
        topo_test  = topo_40
        
    elif args.split_type == 2:
        data_train_val = data_40 + data_190
        topo_train_val = topo_40 + topo_190
        data_test  = data_90
        topo_test  = topo_90
        
    
    if args.pure_type == 0:
        patterns = [data_train_val[i].pattern.numpy() for i in range(len(data_train_val))]
        patterns = np.array(patterns).squeeze()
        
        topos = topo_train_val
        topos = np.array(topos).squeeze()
        
        combined_labels = [f"{pattern}_{topo}" for pattern, topo in zip(patterns, topos)]
        
        output = train_test_split(data_train_val, 
                                topo_train_val, 
                                test_size=0.2, 
                                random_state=SEED,
                                stratify=combined_labels)
    
    elif args.pure_type == 1:
        data_train_val = data_40 + data_90 + data_190
        topo_train_val = topo_40 + topo_90 + topo_190
        
        data_train = []
        topo_train = []
        data_rest  = []
        topo_rest  = []
        
        for i, data_temp in enumerate(data_train_val):
            if data_temp['pattern'] == 0 or data_temp['pattern'] == 1:
                data_train.append(data_temp)
                topo_train.append(topo_train_val[i])
                
            elif data_temp['pattern'] == 2:
                data_rest.append(data_temp)
                topo_rest.append(topo_train_val[i])
            else:
                raise ValueError("Invalid pattern type value")
            
        patterns = [data_train[i].pattern.numpy() for i in range(len(data_train))]
        patterns = np.array(patterns).squeeze()
        
        topos = topo_train
        topos = np.array(topos).squeeze()
        
        combined_labels = [f"{pattern}_{topo}" for pattern, topo in zip(patterns, topos)]
            
        output = train_test_split(data_train, 
                                  topo_train, 
                                  test_size=0.2, 
                                  random_state=SEED,
                                  stratify=combined_labels)
        
        data_test = data_rest
        topo_test = topo_rest
    
    elif args.pure_type == 2:
        if args.split_type == 0:
            data_train_val = data_40 + data_90
            topo_train_val = topo_40 + topo_90
        elif args.split_type == 1:
            data_train_val = data_90 + data_190
            topo_train_val = topo_90 + topo_190
        elif args.split_type == 2:
            data_train_val = data_40 + data_190
            topo_train_val = topo_40 + topo_190
        
        data_train = []
        topo_train = []
        data_rest  = []
        topo_rest  = []
        
        for i, data_temp in enumerate(data_train_val):
            if data_temp['pattern'] == 0 or data_temp['pattern'] == 1:
                data_train.append(data_temp)
                topo_train.append(topo_train_val[i])
                
            elif data_temp['pattern'] == 2:
                data_rest.append(data_temp)
                topo_rest.append(topo_train_val[i])
            else:
                raise ValueError("Invalid pattern type value")
            
        patterns = [data_train[i].pattern.numpy() for i in range(len(data_train))]
        patterns = np.array(patterns).squeeze()
        
        topos = topo_train
        topos = np.array(topos).squeeze()
        
        combined_labels = [f"{pattern}_{topo}" for pattern, topo in zip(patterns, topos)]
            
        output = train_test_split(data_train, 
                                  topo_train, 
                                  test_size=0.2, 
                                  random_state=SEED,
                                  stratify=combined_labels)
        if args.split_type == 0:
            data_test = data_rest + data_190
            topo_test = topo_rest + topo_190
        elif args.split_type == 1:
            data_test = data_rest + data_40
            topo_test = topo_rest + topo_40
        elif args.split_type == 2:
            data_test = data_rest + data_90
            topo_test = topo_rest + topo_90
    else:
        raise ValueError("Invalid pure type value")
    
    return output, data_test, topo_test
