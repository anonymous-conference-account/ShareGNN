from typing import List

import networkx as nx
import numpy as np

from src.Preprocessing.create_splits import splits_from_index_lists
from src.utils.BenchmarkDatasetGeneration.RingTransfer import RingTransfer
from src.Preprocessing.GraphData.GraphData import zinc_to_graph_data
from src.utils.snowflake_generation import Snowflakes, glue_graphs, glue_graphs_edge
import torch_geometric.datasets as tgd

def long_rings(data_size=1200, ring_size=100, seed=764,*args, **kwargs) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    # seed numpy
    np.random.seed(seed)
    while len(graphs) < data_size:
        G = nx.Graph()
        for j in range(0, ring_size):
            G.add_node(j, primary_label=0)
        for j in range(0, ring_size):
            G.add_edge(j % ring_size, (j + 1) % ring_size)
        # permute the Ids of the nodes
        random_permutation = np.random.permutation(ring_size)
        G = nx.relabel_nodes(G, {i: random_permutation[i] for i in range(ring_size)})
        # get a random node and the one on the opposite and the one on 90 degree and 270 and assign random labels from the list {1,2,3,4}
        pos = np.random.randint(0, ring_size)
        node_0 = random_permutation[pos]
        node_1 = random_permutation[(pos + ring_size // 4) % ring_size]
        node_2 = random_permutation[(pos + ring_size // 2) % ring_size]
        node_3 = random_permutation[(pos + 3 * ring_size // 4) % ring_size]
        # randomly shuffle {1,2,3,4} and assign to the nodes
        rand_perm = np.random.permutation([1, 2, 3, 4])
        # change the labels of the nodes
        G.nodes[node_0]['primary_node_labels'] = rand_perm[0]
        G.nodes[node_1]['primary_node_labels'] = rand_perm[1]
        G.nodes[node_2]['primary_node_labels'] = rand_perm[2]
        G.nodes[node_3]['primary_node_labels'] = rand_perm[3]
        # find position of 1 in rand_perm
        pos_one = np.where(rand_perm == 1)[0][0]
        # find label opposite to 1
        pos_opp = (pos_one + 2) % 4
        label_opp = rand_perm[pos_opp]
        if label_opp == 2:
            label = 0
        elif label_opp == 3:
            label = 1
        elif label_opp == 4:
            label = 2
        # get unique label count and append if count for label is smaller than data_size//6
        unique_labels, counts = np.unique(labels, return_counts=True)
        if label not in labels or counts[unique_labels == label] < data_size // 3:
            graphs.append(G)
            labels.append(label)
    # shuffle the graphs and labels
    perm = np.random.permutation(len(graphs))
    graphs = [graphs[i] for i in perm]
    labels = [labels[i] for i in perm]
    return graphs, labels

def even_odd_rings(data_size=1200, ring_size=100, difficulty=1, count=False, seed=764,*args, **kwargs) -> (List[nx.Graph], List[int]):
    """
    Create a benchmark dataset consisting of labeled rings with ring_size nodes and labels.
    The label of the graph is determined by the following:
    - Select the node with label and the node with distance ring_size//2 say x and the ones with distances ring_size//4, ring_size//8, say y_1, y_2 and z_1, z_2
    Now consider the numbers:
    a = 1 + x
    b = y_1 + y_2
    c = z_1 + z_2
    and distinct the cases odd and even. This defines the 8 possible labels of the graphs.
    """
    graphs = []
    labels = []
    # seed numpy
    np.random.seed(seed)
    class_number = 0
    permutation_storage = []
    while len(graphs) < data_size:
        G = nx.Graph()
        label_permutation = np.random.permutation(ring_size)
        for j in range(0, ring_size):
            G.add_node(j, primary_node_labels=label_permutation[j])
        for j in range(0, ring_size):
            G.add_edge(j % ring_size, (j + 1) % ring_size)
        # permute the Ids of the nodes
        random_permutation = np.random.permutation(ring_size)

        # make random permutation start with 0
        r_perm = np.roll(random_permutation, -np.where(random_permutation == 0)[0][0])
        # to list
        r_perm = r_perm.tolist()
        if r_perm not in permutation_storage:
            # add permutation to storage
            permutation_storage.append(r_perm)

            G = nx.relabel_nodes(G, {i: random_permutation[i] for i in range(ring_size)})
            if count:
                class_number = 2
                opposite_nodes = []
                for node in G.nodes(data=True):
                    node_label = node[1]['primary_node_labels']
                    node_id = node[0]
                    pos = np.where(random_permutation == node_id)[0][0]
                    # get opposite node in the ring
                    opposite_node = random_permutation[(pos + ring_size // 2) % ring_size]
                    # get opposite node label in the ring
                    opposite_node_label = G.nodes[opposite_node]['primary_node_labels']
                    # add node_label + opposite_node_label to opposite_nodes
                    opposite_nodes.append(node_label + opposite_node_label)
                # count odd and even entries in opposite_nodes
                odd_count = np.count_nonzero(np.array(opposite_nodes) % 2)
                even_count = len(opposite_nodes) - odd_count
                if odd_count > even_count:
                    label = 1
                else:
                    label = 0
            else:
                # find graph node with label 0
                for node in G.nodes(data=True):
                    if node[1]['primary_node_labels'] == 0:
                        node_0 = node[0]
                        break
                # get index of node_0 in random_permutation
                pos = np.where(random_permutation == node_0)[0][0]
                node_1 = random_permutation[(pos + ring_size // 4) % ring_size]
                node_2 = random_permutation[(pos + ring_size // 2) % ring_size]
                node_3 = random_permutation[(pos - ring_size // 4) % ring_size]
                # get the neighbors of node_0
                node_4 = random_permutation[(pos + 1) % ring_size]
                node_5 = random_permutation[(pos - 1 + ring_size) % ring_size]

                label_node_1 = G.nodes[node_1]['primary_node_labels']
                label_node_2 = G.nodes[node_2]['primary_node_labels']
                label_node_3 = G.nodes[node_3]['primary_node_labels']
                label_node_4 = G.nodes[node_4]['primary_node_labels']
                label_node_5 = G.nodes[node_5]['primary_node_labels']

                # add the labels of the nodes
                a = 0 + label_node_2
                b = label_node_1 + label_node_3
                c = label_node_4 + label_node_5

                if difficulty == 1:
                    label = a % 2
                    class_number = 2
                elif difficulty == 2:
                    label = 2 * (a % 2) + b % 2
                    class_number = 4
                elif difficulty == 3:
                    label = 4 * (a % 2) + 2 * (b % 2) + c % 2
                    class_number = 8

            # get unique label count and append if count for label is smaller than data_size//6
            unique_labels, counts = np.unique(labels, return_counts=True)
            if label not in labels or counts[unique_labels == label] < data_size // class_number:
                graphs.append(G)
                labels.append(label)
    # shuffle the graphs and labels
    perm = np.random.permutation(len(graphs))
    graphs = [graphs[i] for i in perm]
    labels = [labels[i] for i in perm]
    return graphs, labels

def ring_diagonals( data_size=1200, ring_size=100,*args, **kwargs) -> (List[nx.Graph], List[int]):
    """
    Create a dataset of ring graphs with diagonals.
    :param data_size: number of graphs to create
    :param ring_size: number of nodes in each ring
    :return: a list of graphs and a list of labels
    """
    graphs = []
    labels = []
    seed = 16
    np.random.seed(seed)
    class_counter = [0, 0]
    while sum(class_counter) < data_size:
        G = nx.cycle_graph(ring_size)
        # add random 1-dim labels and 3-dim features to nodes and edges
        for node in G.nodes():
            G.nodes[node]['label'] = np.random.randint(0, 2)
            G.nodes[node]['feature'] = [np.random.rand(), np.random.rand(), np.random.rand()]
        for edge in G.edges():
            G[edge[0]][edge[1]]['primary_node_labels'] = np.random.randint(0, 2)
            G[edge[0]][edge[1]]['feature'] = [np.random.rand(), np.random.rand(), np.random.rand()]

        # get two random nodes in the ring and connect them with an edge
        diag_start = np.random.randint(ring_size)
        while True:
            diag_end = np.random.randint(ring_size)
            if diag_end != diag_start:
                break
        # get the distance in the ring between the two nodes
        dist = nx.shortest_path_length(G, diag_start, diag_end)
        G.add_edge(diag_start, diag_end)
        G[diag_start][diag_end]['primary_edge_labels'] = np.random.randint(0, 2)
        G[diag_start][diag_end]['feature'] = [np.random.rand(), np.random.rand(), np.random.rand()]
        # determine the label of the graph G
        # Case 1: Edge Label of the diagonal is 1
        # Case 2: Labels of the two end nodes of the diagonal are the same
        # Case 3: Distance between the two end nodes of the diagonal greater than 25
        # => Then the graph label is 1, else 0
        graph_label = 0
        edge = G.edges[diag_start, diag_end]
        if 'primary_edge_labels' in edge and edge['primary_edge_labels'] == 1:
            graph_label = 1
        elif G.nodes[diag_start]['primary_node_labels'] == G.nodes[diag_end]['primary_node_labels']:
            graph_label = 1
        elif dist > 13:
            graph_label = 1
        if class_counter[graph_label] >= data_size / 2:
            continue
        else:
            class_counter[graph_label] += 1
            labels.append(graph_label)
            graphs.append(G)

    return graphs, labels

def snowflakes(smallest_snowflake=3, largest_snowflake=12, flakes_per_size=100, seed=764, generation_type='binary',*args, **kwargs) -> (List[nx.Graph], List[int]):
    """
    Create a dataset of snowflake graphs.
    """
    return Snowflakes(smallest_snowflake=smallest_snowflake, largest_snowflake=largest_snowflake, flakes_per_size=flakes_per_size, plot=False, seed=seed, generation_type=generation_type)


def counting_rings(data_size=1000, ring_size=3, min_rings=0, max_rings=9, seed=42, *args, **kwargs) -> (List[nx.Graph], List[int]):
    ring_counts = list(range(min_rings, max_rings+1))
    mean_graph_size = max_rings*ring_size
    graphs = []
    labels = []
    for num_rings in ring_counts:
        for graph_id in range(data_size//len(ring_counts)):
            # add num_rings rings to the graph
            rings = [nx.cycle_graph(ring_size) for _ in range(num_rings)]
            # if len(rings) == 0: create a random tree with mean size of mean_graph_size
            if num_rings == 0:
                tree_size = np.random.randint(mean_graph_size -mean_graph_size//2, mean_graph_size + mean_graph_size//2)
                graphs.append(nx.random_tree(tree_size))
                labels.append(0)
            else:
                current_graph = rings[0]
                # iteratively take randomly a node or an edge and glue a new ring to it
                for glueing in range(1, num_rings):
                    rand_int = np.random.randint(0, 2)
                    if rand_int == 0:
                        # get a random node from the current graph
                        rand_node_current = np.random.randint(0, len(current_graph.nodes))
                        rand_node_ring = np.random.randint(0, len(rings[glueing].nodes))
                        current_graph = glue_graphs(current_graph, rings[glueing], rand_node_current, rand_node_ring, plot=False)
                    else:
                        # get a random edge from the current graph
                        rand_edge_current = np.random.randint(0, len(current_graph.edges))
                        rand_edge_ring = np.random.randint(0, len(rings[glueing].edges))
                        current_graph = glue_graphs_edge(current_graph, rings[glueing], rand_edge_current, rand_edge_ring, plot=False)

                # get number of nodes in the current graph
                num_nodes = len(current_graph.nodes)
                # get difference to mean_graph_size
                diff = mean_graph_size - num_nodes
                # if diff > 0 add random nodes to the graph
                nodes_to_add = np.random.randint(num_nodes, mean_graph_size + diff)
                mean_tree_size = 5
                num_trees = nodes_to_add // mean_tree_size
                trees = [nx.random_tree(np.random.randint(1, 2*mean_tree_size - 1)) for _ in range(num_trees)]
                for tree in trees:
                    # get random node from the current graph
                    rand_node_current = np.random.randint(0, len(current_graph.nodes))
                    rand_node_tree = np.random.randint(0, len(tree.nodes))
                    current_graph = glue_graphs(current_graph, tree, rand_node_current, rand_node_tree, plot=False)

                graphs.append(current_graph)
                labels.append(num_rings)
    return graphs, labels




def csl_graphs(*args, **kwargs) -> (List[nx.Graph], List[int]):
    from src.Preprocessing.csl import CSL
    csl = CSL()
    graph_data = csl.get_graphs(with_distances=False)
    return graph_data.graphs, graph_data.graph_labels

def torch_geometric_dataset(name=None,*args, **kwargs) -> (List[nx.Graph], List[int]):
    # get label_path from args
    split_path = kwargs.get("split_path", None)
    # raise error if output_path is None
    if split_path is None:
        raise ValueError("label_path is None")
    if name == "zinc":
        zinc_train = tgd.ZINC(root="tmp/ZINC/", subset=True, split='train')
        zinc_val = tgd.ZINC(root="tmp/ZINC/", subset=True, split='val')
        zinc_test = tgd.ZINC(root="tmp/ZINC/", subset=True, split='test')
        # zinc to networkx
        networkx_graphs = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC")
        # get train, val, test indices from temp path
        train_indices = [[i for i in range(len(zinc_train))]]
        val_indices = [[i + len(zinc_train) for i in range(len(zinc_val))]]
        test_indices = [[i + len(zinc_train) + len(zinc_val) for i in range(len(zinc_test))]]
        splits_from_index_lists(train_indices, val_indices, test_indices, "ZINC", output_path=split_path)
        return networkx_graphs.graphs, networkx_graphs.graph_labels
    elif name == "zinc_full":
        zinc_train = tgd.ZINC(root="tmp/", split='train')
        zinc_val = tgd.ZINC(root="tmp/", split='val')
        zinc_test = tgd.ZINC(root="tmp/", split='test')
        # zinc to networkx
        networkx_graphs = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC")
        return networkx_graphs.graphs, networkx_graphs.graph_labels
    pass

def ring_transfer(data_size=1200, node_dimension=10, ring_size=100, seed=764,*args, **kwargs) -> (List[nx.Graph], List[np.ndarray[float]]):
    return RingTransfer(data_size=data_size, node_dimension=node_dimension, ring_size=ring_size, seed=seed)

def parity_check(data_size=1500, max_size=40, seed=764,*args, **kwargs) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    np.random.seed(seed)
    for i in range(data_size):
        size = np.random.randint(1, max_size + 1)
        G = nx.Graph()
        for j in range(size):
            G.add_node(j, primary_node_labels=0)
        for j in range(size - 1):
            G.add_edge(j, j + 1)
        # create random seqeunce of size size of 0s and 1s
        rand_sequence = np.random.randint(0, 2, size)
        # assign the labels to the nodes
        for j in range(size):
            G.nodes[j]['primary_node_labels'] = rand_sequence[j]
        graphs.append(G)
        # count number of 1s in the sequence
        even = np.count_nonzero(rand_sequence) % 2
        labels.append(even)
    return graphs, labels

def even_pairs(data_size=1500, max_size=40, seed=764,*args, **kwargs) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    np.random.seed(seed)
    for i in range(data_size):
        size = np.random.randint(1, max_size + 1)
        G = nx.Graph()
        for j in range(size):
            G.add_node(j, primary_node_labels=0)
        for j in range(size - 1):
            G.add_edge(j, j + 1)
        # create random seqeunce of size size of 0s and 1s
        rand_sequence = np.random.randint(0, 2, size)
        # assign the labels to the nodes
        for j in range(size):
            G.nodes[j]['primary_node_labels'] = rand_sequence[j]
        graphs.append(G)
        # check wheter first and last node have the same label 0
        valid = rand_sequence[0] == rand_sequence[-1]
        labels.append(valid)
    return graphs, labels

def first_a(data_size=1500, max_size=40, seed=764,*args, **kwargs) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    np.random.seed(seed)
    for i in range(data_size):
        size = np.random.randint(1, max_size + 1)
        G = nx.Graph()
        for j in range(size):
            G.add_node(j, primary_node_labels=0)
        for j in range(size - 1):
            G.add_edge(j, j + 1)
        # create random seqeunce of size size of 0s and 1s
        rand_sequence = np.random.randint(0, 2, size)
        # assign the labels to the nodes
        for j in range(size):
            G.nodes[j]['primary_node_labels'] = rand_sequence[j]
        graphs.append(G)
        # check whether first and last node have the same label 0
        valid = rand_sequence[0]
        labels.append(valid)
    return graphs, labels

def node_classification_test(data_size=1, max_size=1000, num_node_features=1,seed=764,*args, **kwargs) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    np.random.seed(seed)
    for i in range(data_size):
        size = np.random.randint(1, max_size + 1)
        G = nx.Graph()
        for j in range(size):
            G.add_node(j, primary_node_labels=0)
        for j in range(size - 1):
            G.add_edge(j, j + 1)
        # create random seqeunce of size size of 0s and 1s
        rand_sequence = np.random.randint(0, 2, size)
        # assign the labels to the nodes
        for j in range(size):
            G.nodes[j]['primary_node_labels'] = rand_sequence[j]
        graphs.append(G)
        # check wheter first and last node have the same label 0
        valid = rand_sequence[0]
        labels.append(valid)
    return graphs, labels


