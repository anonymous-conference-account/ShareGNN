import networkx as nx

from src.Preprocessing.GraphData.GraphData import get_graph_data


def main():
    # load the graph data
    graph_dbs = ['NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    graph_dbs = ['LongRings100', 'EvenOddRings2_16', 'EvenOddRingsCount16', 'CSL_original', 'Snowflakes']
    outputs = []
    for graph_db in graph_dbs:
        #graph_db = 'Snowflakes'
        data_path = '../../GraphData/DS_all/'
        data_path = '../../Testing/BenchmarkGraphs/'
        distance_path = "../../Testing/Distances/"
        """
        Create Input data, information and labels from the graphs for training and testing
        """
        graph_data = get_graph_data(graph_db, data_path, distance_path, use_labels=True, use_attributes=False)
        # get a list of node numbers sorted by the number of nodes in each graph
        node_numbers = []
        edge_numbers = []
        diameters = []
        for graph in graph_data.graphs:
            node_numbers.append(graph.number_of_nodes())
            edge_numbers.append(graph.number_of_edges())
            # get all connected components
            connected_components = nx.connected_components(graph)
            for connected_component in connected_components:
                # get the diameter of the connected component
                subgraph = graph.subgraph(connected_component)
                diameter = nx.diameter(subgraph)
                diameters.append(diameter)
        # num graphs
        print(f'Num Graphs: {len(graph_data.graphs)}')

        # round to 1 decimal
        node_numbers = [round(x, 1) for x in node_numbers]

        print(f'Max nodes / avg nodes / min nodes: {max(node_numbers)} & {round(sum(node_numbers) / len(node_numbers), 1)} & {min(node_numbers)}')
        print(f'Max edges / avg edges / min edges: {max(edge_numbers)} & {round(sum(edge_numbers) / len(edge_numbers), 1)} & {min(edge_numbers)}')
        print(f'Max diameter / avg diameter / min diameter: {max(diameters)} & {round(sum(diameters) / len(diameters), 1)} & {min(diameters)}')

        print(f'Num node labels:{graph_data.node_labels["primary"].num_unique_node_labels}')
        node_numbers.sort()
        # ge the 0.6 and 0.9 median
        print(f'Median 0.6: {node_numbers[int(0.6 * len(node_numbers))]}')
        print(f'Median 0.9: {node_numbers[int(0.9 * len(node_numbers))]}')

        # table output
        outputs.append(f'{graph_db} & {len(graph_data.graphs)} & {max(node_numbers)} & {round(sum(node_numbers) / len(node_numbers), 1)} & {min(node_numbers)} & {max(edge_numbers)} & {round(sum(edge_numbers) / len(edge_numbers), 1)} & {min(edge_numbers)} & {max(diameters)} & {round(sum(diameters) / len(diameters), 1)} & {min(diameters)} & {graph_data.node_labels["primary"].num_unique_node_labels} & {graph_data.num_classes} \\\\')

    for output in outputs:
        print(output)

if __name__ == "__main__":
    main()