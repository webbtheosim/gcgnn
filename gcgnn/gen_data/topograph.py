import networkx as nx

def gen_linear(n):
    return nx.path_graph(n)

def gen_cyclic(n):
    return nx.cycle_graph(n)

def gen_comb(num_backbone, gap_branch, len_branch):
    assert gap_branch >= 1, "gap_branch should be greater than or equal to 2"
    
    num_branch = (num_backbone - 1) // gap_branch
    assert num_branch >= 3, "num_branch should be greater than or equal to 3"

    G = nx.path_graph(num_backbone)

    for i in range(1, num_backbone + 1 - gap_branch, gap_branch):
        branch = nx.path_graph(len_branch)
        branch = nx.relabel_nodes(branch, {j: i*num_backbone + j for j in range(len_branch)})
        G = nx.union(G, branch)
        G.add_edge(i, i*num_backbone)
    return G

def gen_star(num_arm, len_branch):
    assert num_arm >= 3, "num_arm should be greater than or equal to 3"
    assert len_branch >= 2, "len_branch should be greater than or equal to 2"
    
    G = nx.Graph()
    G.add_node(0)

    for i in range(1, num_arm + 1):
        arm = nx.path_graph(len_branch)
        arm = nx.relabel_nodes(arm, {j: i*len_branch + j for j in range(len_branch)})
        G = nx.union(G, arm)
        G.add_edge(0, i*len_branch)
    return G

def gen_astar(num_arm, len_branch, minus_num_short, minus_len_short):
    assert num_arm >= 3, "num_arm should be greater than or equal to 3"
    assert 0 < minus_num_short < num_arm, "minus_num_short should be in (0, num_arm)"
    assert 0 < minus_len_short < len_branch, "minus_len_short should be in (0, len_branch)"

    G = nx.Graph()
    G.add_node(0)

    for i in range(1, num_arm + 1):
        len_branch_temp = (len_branch - minus_len_short) if i <= num_arm - minus_num_short else len_branch
        arm = nx.path_graph(len_branch_temp)
        arm = nx.relabel_nodes(arm, {j: i*len_branch_temp + j for j in range(len_branch_temp)})
        G = nx.union(G, arm)
        G.add_edge(0, i*len_branch_temp)

    # note: palm tree or omega branch is one class of asymmetric star
    return G

def gen_branch(num_backbone, num_branch, len_branch):
    assert num_branch >= 2, "num_branch should be greater than or equakl to 2"

    G = nx.Graph()
    G = nx.path_graph(num_backbone)

    for i in range(1, num_branch+1):
        branch = nx.path_graph(len_branch)
        branch = nx.relabel_nodes(branch, {j: i*num_backbone + j for j in range(len_branch)})
        G = nx.union(G, branch)
        G.add_edge(0, i*num_backbone)

    for i in range(1, num_branch+1):
        branch = nx.path_graph(len_branch)
        branch = nx.relabel_nodes(branch, {j: (i+num_branch)*num_backbone + j for j in range(len_branch)})
        G = nx.union(G, branch)
        G.add_edge(num_backbone-1, (i+num_branch)*num_backbone)

    # note: also called pompom
    return G


def gen_abranch(num_backbone, num_branch, len_branch, minus_num_short, minus_len_short):
    assert num_branch >= 2, "num_branch should be greater than or equal to 2"
    assert 0 < minus_num_short < num_branch-1, "minus_num_short should be in (1, num_branch)"
    assert 0 < minus_len_short < len_branch, "minus_len_short should be in (0, len_branch)"

    G = nx.Graph()
    G = nx.path_graph(num_backbone)

    for i in range(1, num_branch+1):
        branch = nx.path_graph(len_branch)
        branch = nx.relabel_nodes(branch, {j: i*num_backbone + j for j in range(len_branch)})
        G = nx.union(G, branch)
        G.add_edge(0, i*num_backbone)

    num_short_branch = num_branch - minus_num_short
    len_short_branch = len_branch - minus_len_short

    for i in range(1, num_short_branch+1):
        branch = nx.path_graph(len_short_branch)
        branch = nx.relabel_nodes(branch, {j: (i+num_branch)*num_backbone + j for j in range(len_short_branch)})
        G = nx.union(G, branch)
        G.add_edge(num_backbone-1, (i+num_branch)*num_backbone)

    return G


def gen_dendrimer(num_branch, generation):
    assert generation >= 1, "generation should be greater than or equal to 1"
    assert num_branch >= 2, "num_branch should be greater than or equal to 2"

    def _add_generation(G, nodes, num_branch, generation):
        if generation == 0:
            return G

        new_nodes = []
        for node in nodes:
            for _ in range(num_branch):
                new_node = max(G.nodes) + 1
                G.add_edge(node, new_node)
                new_nodes.append(new_node)
        return _add_generation(G, new_nodes, num_branch, generation - 1)


    
    G = nx.Graph()
    G.add_node(0)

    G = _add_generation(G, [0], num_branch, generation)

    return G


def gen_conical_bottlebrush(num_backbone, gap_branch, max_len_branch):
    assert gap_branch in [0, 1], "gap_branch should be either 0 or 1"
    
    G = nx.Graph()
    # Add the backbone
    backbone_nodes = list(range(num_backbone))
    G.add_nodes_from(backbone_nodes)
    nx.add_path(G, backbone_nodes)

    # Add branches
    node_count = num_backbone
    for i in range(num_backbone):
        branch_length = max_len_branch - (max_len_branch * i // num_backbone)
        if branch_length > 0:
            # Nodes for the branch
            branch_nodes = list(range(node_count, node_count + branch_length))
            node_count += branch_length
            # Add the branch to the graph
            G.add_nodes_from(branch_nodes)
            nx.add_path(G, branch_nodes)
            # Connect the branch to the backbone
            G.add_edge(i, branch_nodes[0])

        if gap_branch == 1 and i < num_backbone - 1:
            G.add_edge(i, i + 1)

    return G


def gen_bowtie_bottlebrush(num_backbone, gap_branch, max_len_branch):
    assert gap_branch in [0, 1], "gap_branch should be either 0 or 1"
    
    G = nx.Graph()
    # Add the backbone
    backbone_nodes = list(range(num_backbone))
    G.add_nodes_from(backbone_nodes)
    nx.add_path(G, backbone_nodes)

    # Add branches
    node_count = num_backbone
    third = num_backbone // 3  # One third of the backbone length

    for i in range(num_backbone):
        # Determine branch length for each section
        if i < third:  # First third: decreasing length
            branch_length = max_len_branch - (max_len_branch * i // third)
        elif i < 2 * third:  # Middle third: constant length
            branch_length = max_len_branch
        else:  # Last third: increasing length (mirrored from first third)
            branch_length = max_len_branch - (max_len_branch * (num_backbone - 1 - i) // third)

        # Nodes for the branch
        if branch_length > 0:
            branch_nodes = list(range(node_count, node_count + branch_length))
            node_count += branch_length
            # Add the branch to the graph
            G.add_nodes_from(branch_nodes)
            nx.add_path(G, branch_nodes)
            # Connect the branch to the backbone
            G.add_edge(i, branch_nodes[0])

        if gap_branch == 1 and i < num_backbone - 1:
            G.add_edge(i, i + 1)

    return G