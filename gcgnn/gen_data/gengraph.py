from .topograph import gen_linear
from .topograph import gen_cyclic
from .topograph import gen_comb
from .topograph import gen_star
from .topograph import gen_astar
from .topograph import gen_branch
from .topograph import gen_dendrimer
from .topograph import gen_conical_bottlebrush
from .topograph import gen_bowtie_bottlebrush
from .graphutils import relabel
from .graphutils import get_desc

import copy
import numpy as np
import networkx as nx

def run_linear(min_bead, max_bead):
    """Generate linear topologies"""
    graph, label, graph_desc, poly_label = [], [], [], []

    for num_backbone in range(min_bead, max_bead+1):
        G = gen_linear(num_backbone)
        G = relabel(G)
        graph.append(G)
        label.append('linear')
        graph_desc.append(get_desc(G))
        poly_label.append([num_backbone])
        
    return (graph, label, graph_desc, poly_label)


def run_cyclic(min_bead, max_bead):
    """Generate cyclic topologies"""
    graph, label, graph_desc, poly_label = [], [], [], []

    for num_backbone in range(min_bead, max_bead+1):
        G = gen_cyclic(num_backbone)
        G = relabel(G)
        graph.append(G)
        label.append('cyclic')
        graph_desc.append(get_desc(G))
        poly_label.append([num_backbone])
        
    return (graph, label, graph_desc, poly_label)


def run_comb(min_bead, max_bead, backbone_range, gap_range, len_branch_range):
    """Generate comb and bottlebrush topologies"""
    graph, label, graph_desc, poly_label = [], [], [], []

    for num_backbone in backbone_range:
        for gap in gap_range:
            for len_branch in len_branch_range:
                try:
                    num_node_approx = num_backbone + (num_backbone // gap) * len_branch
                    
                    if (num_node_approx > max_bead) or (num_node_approx < min_bead):
                        continue
                    else:
                        G = gen_comb(num_backbone, gap, len_branch)

                    if (G.number_of_nodes() > max_bead) or (G.number_of_nodes() < min_bead):
                        continue
                    else:
                        G = relabel(G)
                        graph.append(G)
                        if gap < 2:
                            label.append('bottlebrush_normal')
                        else:
                            label.append('comb')
                        graph_desc.append(get_desc(G))
                        poly_label.append([num_backbone, gap, len_branch])
                except:
                    continue
    return (graph, label, graph_desc, poly_label)


def run_branch(min_bead, max_bead, backbone_range, num_branch_range, len_branch_range):
    """Generate alpha omega branch topologies"""
    graph, label, graph_desc, poly_label = [], [], [], []
    
    for num_backbone in backbone_range:
        for num_branch in num_branch_range:
            for len_branch in len_branch_range:
                try:
                    num_node_approx = num_backbone + num_branch * len_branch * 2
                    
                    if (num_node_approx > max_bead) or (num_node_approx < min_bead):
                        continue
                    else:
                        G = gen_branch(num_backbone, num_branch, len_branch)
                    if (G.number_of_nodes() > max_bead) or (G.number_of_nodes() < min_bead):
                        continue
                    else:
                        G = relabel(G)
                        graph.append(G)
                        label.append('branch')
                        graph_desc.append(get_desc(G))     
                        poly_label.append([num_backbone, num_branch, len_branch])
                except:
                    continue
    return (graph, label, graph_desc, poly_label)


def run_star(min_bead, max_bead, num_branch_range, len_branch_range):
    """Generate star topologies"""
    graph, label, graph_desc, poly_label = [], [], [], []
    
    break_flag = False
    for num_branch in num_branch_range:
        break_flag = False
        for len_branch in len_branch_range:
            try:
                num_node_approx = 1 + num_branch * len_branch
                if (num_node_approx > max_bead) or (num_node_approx < min_bead):
                    continue
                else:
                    G = gen_star(num_branch, len_branch)
                if (G.number_of_nodes() > max_bead) or (G.number_of_nodes() < min_bead):
                    continue
                else:
                    G = relabel(G)
                    graph.append(G)
                    label.append('star')
                    graph_desc.append(get_desc(G))     
                    poly_label.append([num_branch, len_branch])
            except:
                continue
        if break_flag:
            continue
            
    return (graph, label, graph_desc, poly_label)


def run_astar(min_bead, max_bead, num_branch_range, len_branch_range, ratio_num, ratio_len):
    """Generate asymmetric star topologies"""
    graph, label, graph_desc, poly_label = [], [], [], []
    
    for num_branch in num_branch_range:
        break_flag = False
        for len_branch in len_branch_range:
            if break_flag: 
                break
            for minus_num_short in ratio_num:
                if break_flag:
                    break
                for minus_len_short in ratio_len: 
                    try:
                        num_node_approx = 1 + num_branch*len_branch-int(num_branch*minus_num_short) * (len_branch-int(len_branch*minus_len_short))
                        if (num_node_approx > max_bead) or (num_node_approx < min_bead):
                            continue
                        else:
                            G = gen_astar(num_branch, len_branch, 
                                          int(num_branch*minus_num_short), int(len_branch*minus_len_short))
                        if (G.number_of_nodes() > max_bead) or (G.number_of_nodes() < min_bead):
                            continue
                        else:
                            G = relabel(G)
                            graph.append(G)
                            label.append('stara')
                            graph_desc.append(get_desc(G))     
                            poly_label.append([num_branch, len_branch, int(num_branch*minus_num_short), 
                                               int(len_branch*minus_len_short)])
                    except:
                        continue
        if break_flag:
            continue


def trim_dendrimer(G, target_nodes):
    """Trim dendrimer to target number of nodes"""
    assert target_nodes < len(G.nodes), "target_nodes should be less than the number of nodes in the dendrimer"
    G_trim = copy.deepcopy(G)

    while len(G_trim.nodes) > target_nodes:
        leaf_nodes = [node for node, degree in G_trim.degree if degree == 1]
        node = np.random.choice(leaf_nodes)
        G_trim.remove_node(node)

    return G_trim


def generate_unique_trimmed_dendrimers(num_branch, generation, target_nodes, num_dendrimers, max_attempts=1000):
    """Generate unique dendrimers"""
    dendrimers = []
    attempts = 0
    while len(dendrimers) < num_dendrimers and attempts < max_attempts:
        G = gen_dendrimer(num_branch, generation)
        if len(G.nodes) > target_nodes:
            G_trim = trim_dendrimer(G, target_nodes)
            if not any(nx.is_isomorphic(G_trim, G_existing) for G_existing in dendrimers):
                dendrimers.append(G_trim)
        attempts += 1

    if attempts == max_attempts:
        print("Maximum number of attempts reached. Returning dendrimers generated so far.")
    return dendrimers
    
    
def run_dendrimer(min_bead, max_bead, num_branch, generation, num_dendrimers):
    """Generate dendrimer topologies"""
    graph, label, graph_desc, poly_label = [], [], [], []
     
    if sum([num_branch**i for i in range(generation+1)]) < min_bead:
        raise Exception("Too few nodes...")
    
    for i in range(min_bead, max_bead+1):
        Gs = generate_unique_trimmed_dendrimers(num_branch, generation, i, num_dendrimers, max_attempts=100)
        for j in range(len(Gs)):
            G = Gs[j]
            G = relabel(G)
            graph.append(G)
            poly_label.append([num_branch, generation,i,j])
            label.append('dendrimer')
            graph_desc.append(get_desc(G))

    return (graph, label, graph_desc, poly_label)


def run_conical_bottlebrush(min_bead, max_bead, backbone_range, gap_range, len_branch_range):
    """Generate conical bottlebrush topologies"""
    graph, label, graph_desc, poly_label = [], [], [], []

    for num_backbone in backbone_range:
        for gap in gap_range:
            for len_branch in len_branch_range:
                try:
                    G = gen_conical_bottlebrush(num_backbone, gap, len_branch)
                    if (G.number_of_nodes() > max_bead) or (G.number_of_nodes() < min_bead):
                        continue
                    else:
                        G = relabel(G)
                        graph.append(G)
                        label.append('bottlebrush_conical')
                        graph_desc.append(get_desc(G))
                        poly_label.append([num_backbone, gap, len_branch])
                except:
                    continue
    return (graph, label, graph_desc, poly_label)


def run_bowtie_bottlebrush(min_bead, max_bead, backbone_range, gap_range, len_branch_range):
    """Generate bowtie bottlebrush topologies"""
    graph, label, graph_desc, poly_label = [], [], [], []

    for num_backbone in backbone_range:
        for gap in gap_range:
            for len_branch in len_branch_range:
                try:
                    G = gen_bowtie_bottlebrush(num_backbone, gap, len_branch)
                    if (G.number_of_nodes() > max_bead) or (G.number_of_nodes() < min_bead):
                        continue
                    else:
                        G = relabel(G)
                        graph.append(G)
                        label.append('bottlebrush_bowtie')
                        graph_desc.append(get_desc(G))
                        poly_label.append([num_backbone, gap, len_branch])
                except:
                    continue
    return (graph, label, graph_desc, poly_label)
