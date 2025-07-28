import networkx as nx


class SubgraphTree1(nx.Graph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.add_nodes_from([0, 1, 2, 3])
        self.add_edges_from([(0, 1), (0, 2), (0, 3)])


class SubgraphTree2(nx.Graph):
    def __init__(self, **attr):
        super().__init__(**attr)
        self.add_nodes_from([0, 1, 2, 3, 4])
        self.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4)])
