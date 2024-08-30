import pickle
import numpy as np

def str2num(x):
    """ Convert string to solvophobicity parameter """
    if x == 1:
        y = 0.16
    elif x == 0:
        y = 1.0
    else:
        y = 0
    return y


def preprocess_topo(topo):
    """ Combine certain topologies """
    topo[topo == 'stara'] = 'star'
    topo[topo == 'bottlebrush'] = 'comb'
    return topo


def load_graph_rg_data(file_path):
    """ Load graph data and labels """
    with open(file_path, "rb") as handle:
        graph    = pickle.load(handle)
        topo     = pickle.load(handle)
        desc     = pickle.load(handle)
        meta     = pickle.load(handle)
        mode     = pickle.load(handle)
        rg2_mean = pickle.load(handle)
        rg2_var  = pickle.load(handle)
    return graph, topo, rg2_mean, rg2_var


def shuffle_data(x, y):
    """ Shuffle data for training"""
    for x_temp, y_temp in zip(x, y):
        np.random.RandomState(42).shuffle(x_temp)
        np.random.RandomState(42).shuffle(y_temp)