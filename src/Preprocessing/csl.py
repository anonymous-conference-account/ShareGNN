import networkx as nx
import torch
from torch_geometric.datasets import GNNBenchmarkDataset

from src.utils import EdgeLabeling
from src.Preprocessing import NodeLabeling
from src.Preprocessing.GraphData.GraphData import GraphData


class CSL:
    def __init__(self):
        self.data = GNNBenchmarkDataset("tmp/", 'CSL')._data

    def get_graphs(self, with_distances=True):
        nx_graphs = []
        original_source = -1
        # iterate over edge_index in self.data and add edges to nx_graph
        for edge in self.data.edge_index.T:
            source = edge[0].item()
            target = edge[1].item()
            if nx_graphs == [] or source < original_source:
                nx_graphs.append(nx.Graph())
            original_source = source
            nx_graphs[-1].add_edge(source, target)
        labels = [x.item() for x in self.data["y"]]

        self.graphs = GraphData()
        self.graphs.graph_labels = labels
        self.graphs.graphs = nx_graphs
        self.graphs.num_graphs = len(nx_graphs)
        self.graphs.num_classes = len(set(labels))
        self.graphs.name = "CSL"
        self.graphs.input_data = [torch.ones(g.number_of_nodes()).double() for g in nx_graphs]
        if with_distances:
            self.graphs.distance_list = []
            for graph in nx_graphs:
                self.graphs.distance_list.append(dict(nx.all_pairs_shortest_path_length(graph, cutoff=6)))

        # get one hot labels from graph labels
        self.graphs.output_data = torch.nn.functional.one_hot(torch.tensor(labels)).double()
        # set the labels
        self.graphs.add_node_labels(node_labeling_name='primary', node_labeling_method=NodeLabeling.standard_node_labeling)
        self.graphs.add_edge_labels(edge_labeling_name='primary', edge_labeling_method=EdgeLabeling.standard_edge_labeling)
        # add wl_0, wl_1 labels
        #self.graphs.add_node_labels(node_labeling_name='wl_0', node_labeling_method=NodeLabeling.degree_node_labeling)
        #self.graphs.add_node_labels(node_labeling_name='wl_1', node_labeling_method=NodeLabeling.weisfeiler_lehman_node_labeling, max_iterations=1)
        return self.graphs


def main():
    csl = CSL()
    csl.get_graphs()

if __name__ == "__main__":
    main()