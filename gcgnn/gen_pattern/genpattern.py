import networkx as nx
import numpy as np

def process_random(G_in, seed=0):
    """Randomly assign node types"""
    G = G_in.copy()
    l = nx.number_of_nodes(G)
    n_a = int(l / 2)
    n_b = l - n_a
    seed2 = np.random.RandomState(seed).randint(0, 10000, size=l)
    for node in range(nx.number_of_nodes(G)):
        G.nodes[node]['type'] = np.random.RandomState(seed2[node]).choice([0, 1], size=1, p=[n_a/l, n_b/l])[0]
    return G

def process_dendrimer(G_in, desc, mode='0'):
    """Assign node types for dendrimer"""
    G = G_in.copy()
    n_branch = desc[0]
    n_gen = desc[1]
    assert mode in [0, 1, 2, 3, 4, 5], "Must be 0, 1, 2, 3"
    ecc = nx.eccentricity(G)
    center_node = min(ecc, key=ecc.get)
    assert center_node == 0, "Must be 0, preprocess error"
    labels = {center_node: 0}
    current_iteration = 1
    
    while True:
        neighbors_to_label = []
        for node in labels:
            neighbors_to_label.extend(list(G.neighbors(node)))
        neighbors_to_label = [node for node in neighbors_to_label if node not in labels]
        if not neighbors_to_label:
            break
        for node in neighbors_to_label:
            labels[node] = current_iteration
        current_iteration += 1

    for node, iteration_label in labels.items():
        if mode == 0:
            G.nodes[node]['type'] = iteration_label % 2
        elif mode == 1:
            G.nodes[node]['type'] = (iteration_label + 1) % 2
        elif mode == 2:
            G.nodes[node]['type'] = 0 if iteration_label < n_gen else 1
        elif mode == 3:
            G.nodes[node]['type'] = 1 if iteration_label < n_gen else 0
        elif mode == 4:
            G.nodes[node]['type'] = 0
        elif mode == 5:
            G.nodes[node]['type'] = 1

    return G

def process_comb(G_in, desc, back_mode=0, branch_mode=2):
    """Assign node types for comb polymer"""
    G = G_in.copy()
    assert back_mode in [0, 1, 2, 3], "Must be 0, 1, 2, 3"
    assert branch_mode in [0, 1, 2, 3], "Must be 0, 1, 2, 3"
    
    for node in range(desc[0]):
        if back_mode == 0:
            G.nodes[node]['type'] = 0
        elif back_mode == 1:
            G.nodes[node]['type'] = 1
        elif back_mode == 2:
            G.nodes[node]['type'] = node % 2
        elif back_mode == 3:
            G.nodes[node]['type'] = 0 if node < desc[0]/2 else 1
    
    a = desc[0]
    b = desc[0]+desc[2]
    count = 0
    
    while b <= nx.number_of_nodes(G):
        for node in range(a, b):
            if branch_mode == 0:
                G.nodes[node]['type'] = count % 2
            elif branch_mode == 1:
                G.nodes[node]['type'] = 1-G.nodes[0]['type']
            elif branch_mode == 2:
                G.nodes[node]['type'] = G.nodes[0]['type']
            elif branch_mode == 3:
                G.nodes[node]['type'] = node % 2
        a = b
        b = b + desc[2]
        count += 1

    return G

def process_branch(G_in, desc, back_mode=0, branch_mode=2):
    """Assign node types for alpha omega branch polymer"""
    G = G_in.copy()
    n_backbone = desc[0]
    n_branch = desc[1]
    l_branch = desc[2]
    
    assert back_mode in [0, 1, 2, 3], "Must be 0, 1, 2, 3, 4"
    assert branch_mode in [0, 1, 2, 3, 4], "Must be 0, 1, 2, 3"
    
    for node in range(desc[0]): #backbone
        if back_mode == 0:
            G.nodes[node]['type'] = 0
        elif back_mode == 1:
            G.nodes[node]['type'] = 1
        elif back_mode == 2:
            G.nodes[node]['type'] = node % 2
        elif back_mode == 3:
            G.nodes[node]['type'] = 0 if node < desc[0]/2 else 1
    
    a = desc[0]
    b = desc[0]+desc[2]
    count = 0
    
    while b <= nx.number_of_nodes(G):
        for node in range(a, b):
            if branch_mode == 0:
                G.nodes[node]['type'] = count % 2
            elif branch_mode == 1:
                G.nodes[node]['type'] = 1 - G.nodes[0]['type']
            elif branch_mode == 2: 
                G.nodes[node]['type'] = G.nodes[0]['type']
            elif branch_mode == 3:
                G.nodes[node]['type'] = node % 2
            elif branch_mode == 4:
                G.nodes[node]['type'] = 0 if node < n_branch * l_branch + n_backbone else 1
        a = b
        b = b + desc[2]
        count += 1

    return G

def process_star(G_in, desc, mode='0'):
    """Assign node types for star polymer"""
    G = G_in.copy()
    n_arm = desc[0]
    l_arm = desc[1]
    
    assert mode in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "Must be 0~9"
    center_node = 0
    labels = {center_node: 0}
    current_iteration = 1

    while True:
        neighbors_to_label = []
        for node in labels:
            neighbors_to_label.extend(list(G.neighbors(node)))
        neighbors_to_label = [node for node in neighbors_to_label if node not in labels]

        if not neighbors_to_label:
            break

        for node in neighbors_to_label:
            labels[node] = current_iteration

        current_iteration += 1

    for node, iteration_label in labels.items():
        if mode == 0:
            G.nodes[node]['type'] = iteration_label % 2
        elif mode == 1:
            G.nodes[node]['type'] = (iteration_label + 1) % 2
        elif mode == 2:
            G.nodes[node]['type'] = 0 if iteration_label < l_arm else 1
        elif mode == 3:
            G.nodes[node]['type'] = 1 if iteration_label < l_arm else 0
        elif mode == 4:
            G.nodes[node]['type'] = 0 if node < 1 + int(n_arm/2) * l_arm else 1
        elif mode == 5:
            G.nodes[node]['type'] = 1 if node < 1 + int(n_arm/2) * l_arm else 0
        elif mode == 6:
            G.nodes[node]['type'] = 1 if (node -1) % l_arm < int(l_arm / 2) else 0
            G.nodes[0]['type'] = 1
        elif mode == 7:
            G.nodes[node]['type'] = 0 if (node -1) % l_arm < int(l_arm / 2) else 1
            G.nodes[0]['type'] = 0
        elif mode == 8:
            G.nodes[node]['type'] = 0 
        elif mode == 9:
            G.nodes[node]['type'] = 1

    return G


def process_stara(G_in, desc, mode='0'):
    """Assign node types for asymmetric star polymer """
    G = G_in.copy()
    n_arm = desc[0]
    l_arm = desc[1]
    n_short = desc[0] - desc[2]
    l_short = desc[1] - desc[3]
    
    assert mode in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "Must be 0~8"
    center_node = 0
    labels = {center_node: 0}
    current_iteration = 1

    while True:
        neighbors_to_label = []
        for node in labels:
            neighbors_to_label.extend(list(G.neighbors(node)))
        neighbors_to_label = [node for node in neighbors_to_label if node not in labels]
        if not neighbors_to_label:
            break
        for node in neighbors_to_label:
            labels[node] = current_iteration
        current_iteration += 1

    for node, iteration_label in labels.items():
        if mode == 0:
            G.nodes[node]['type'] = iteration_label % 2
        elif mode == 1:
            G.nodes[node]['type'] = (iteration_label + 1) % 2
        elif mode == 2:
            G.nodes[node]['type'] = 0 if node < n_short * l_short + 1 else 1
        elif mode == 3:
            G.nodes[node]['type'] = 1 if node < n_short * l_short + 1 else 0
        elif mode == 4:
            if node == 0:
                G.nodes[node]['type'] = 0
            elif node < n_short * l_short and (node - 1) % l_short < int(l_short / 2):
                G.nodes[node]['type'] = 0
            elif node > n_short * l_short and (node - n_short * l_short - 1) % l_arm < int(l_arm / 2):
                G.nodes[node]['type'] = 0
            else:
                G.nodes[node]['type'] = 1
        elif mode == 5:
            if node == 0:
                G.nodes[node]['type'] = 1
            elif node < n_short * l_short and (node - 1) % l_short < int(l_short / 2):
                G.nodes[node]['type'] = 1
            elif node > n_short * l_short and (node - n_short * l_short - 1) % l_arm < int(l_arm / 2):
                G.nodes[node]['type'] = 1
            else:
                G.nodes[node]['type'] = 0
        elif mode == 6:
            G.nodes[node]['type'] = 1 if (node -1) % l_arm < int(l_arm / 2) else 0
            G.nodes[0]['type'] = 1
        elif mode == 7:
            G.nodes[node]['type'] = 0 if (node -1) % l_arm < int(l_arm / 2) else 1
            G.nodes[0]['type'] = 0
        elif mode == 8:
            G.nodes[node]['type'] = 0 
        elif mode == 9:
            G.nodes[node]['type'] = 1
                
    return G

def process_linear(G_in, desc, mode='0'):
    """Assign node types for linear polymer"""
    G = G_in.copy()
    assert mode in [0, 1, 2, 3], "Must be 0~8"
    l = desc[0]
    
    if mode == 0:
        for node in range(nx.number_of_nodes(G)):
            G.nodes[node]['type'] = 0 if node < int(l / 2) else 1
    elif mode == 1:
        for node in range(nx.number_of_nodes(G)):
            G.nodes[node]['type'] = node % 2
    elif mode == 2:
        for node in range(nx.number_of_nodes(G)):
            G.nodes[node]['type'] = 0
    elif mode == 3:
        for node in range(nx.number_of_nodes(G)):
            G.nodes[node]['type'] = 1
                
    return G


def process_cyclic(G_in, desc, mode='0'):
    """Assign node types for cyclic polymer"""
    G = G_in.copy()
    assert mode in [0, 1, 2, 3], "Must be 0~8"
    l = desc[0]
    
    if mode == 0:
        for node in range(nx.number_of_nodes(G)):
            G.nodes[node]['type'] = 0 if node < int(l / 2) else 1
    elif mode == 1:
        for node in range(nx.number_of_nodes(G)):
            G.nodes[node]['type'] = node % 2
    elif mode == 2:
        for node in range(nx.number_of_nodes(G)):
            G.nodes[node]['type'] = 0
    elif mode == 3:
        for node in range(nx.number_of_nodes(G)):
            G.nodes[node]['type'] = 1
                
    return G

def pattern_dendrimer(graph, poly_label):
    """Generate new dendrimer chemical patterns"""
    G_dendrimer = graph
    spec_dendrimer = poly_label
    
    G_dendrimer_new = []
    for mode in range(6):
        G_temp = process_dendrimer(G_dendrimer, spec_dendrimer, mode=mode)
        G_dendrimer_new.append(G_temp)
        
    for seed in range(19):
        G_temp = process_random(G_dendrimer, seed=seed)
        G_dendrimer_new.append(G_temp)
        
    return G_dendrimer_new


def pattern_comb(graph, poly_label):
    """Generate new comb chemical patterns"""
    G_comb = graph
    spec_comb = poly_label
    
    G_comb_new = []
    for back_mode in range(4):
        for branch_mode in range(4):
            G_temp = process_comb(G_comb, spec_comb, back_mode=back_mode, branch_mode=branch_mode)
            G_comb_new.append(G_temp)
        
    for seed in range(9):
        G_temp = process_random(G_comb, seed=seed)
        G_comb_new.append(G_temp)
    return G_comb_new


def pattern_branch(graph, poly_label):
    """Generate new alpha omega branch chemical patterns"""
    G_branch = graph
    spec_branch = poly_label
    
    G_branch_new = []
    for back_mode in range(4):
        for branch_mode in range(5):
            G_temp = process_branch(G_branch, spec_branch, back_mode=back_mode, branch_mode=branch_mode)
            G_branch_new.append(G_temp)
        
    for seed in range(5):
        G_temp = process_random(G_branch, seed=seed)
        G_branch_new.append(G_temp)
    return G_branch_new


def pattern_star(graph, poly_label):
    """Generate new star chemical patterns"""
    G_star = graph
    spec_star = poly_label
    
    G_star_new = []
    for mode in range(10):
        G_temp = process_star(G_star, spec_star, mode=mode)
        G_star_new.append(G_temp)
        
    for seed in range(15):
        G_temp = process_random(G_star, seed=seed)
        G_star_new.append(G_temp)
        
    return G_star_new


def pattern_stara(graph, poly_label):
    """Generate new asymmetric star chemical patterns"""
    G_stara = graph
    spec_stara = poly_label
    
    G_stara_new = []
    for mode in range(10):
        G_temp = process_stara(G_stara, spec_stara, mode=mode)
        G_stara_new.append(G_temp)
        
    for seed in range(15):
        G_temp = process_random(G_stara, seed=seed)
        G_stara_new.append(G_temp)
        
    return G_stara_new


def pattern_linear(graph, poly_label):
    """Generate new linear chemical patterns"""
    G_linear = graph
    spec_linear = poly_label
    
    G_linear_new = []
    for mode in range(4):
        G_temp = process_linear(G_linear, spec_linear, mode=mode)
        G_linear_new.append(G_temp)
        
    for seed in range(21):
        G_temp = process_random(G_linear, seed=seed)
        G_linear_new.append(G_temp)
        
    return G_linear_new


def pattern_cyclic(graph, poly_label):
    """Generate new cyclic chemical patterns"""
    G_cyclic = graph
    spec_cyclic = poly_label
    
    G_cyclic_new = []
    for mode in range(4):
        G_temp = process_cyclic(G_cyclic, spec_cyclic, mode=mode)
        G_cyclic_new.append(G_temp)
        
    for seed in range(21):
        G_temp = process_random(G_cyclic, seed=seed)
        G_cyclic_new.append(G_temp)
        
    return G_cyclic_new