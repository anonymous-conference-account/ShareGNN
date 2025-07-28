import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.utils.GraphData import GraphData, get_graph_data
from src.utils.snowflake_generation import M0, M1, M3


def draw_graph(graph_data: GraphData, graph_id, ax, node_size=50, edge_color='black',
               edge_width=0.5, draw_type='circle'):
    graph = graph_data.graphs[graph_id]

    # draw the graph
    # root node is the one with label 0
    root_node = None
    for node in graph.nodes():
        if graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
            root_node = node
            break

    node_labels = {}
    for node in graph.nodes():
        key = int(node)
        value = str(graph_data.node_labels['primary'].node_labels[graph_id][node])
        node_labels[key] = f'{value}'
    edge_labels = {}
    for (key1, key2, value) in graph.edges(data=True):
        if "label" in value and len(value["label"]) > 0:
            edge_labels[(key1, key2)] = int(value["label"])
        else:
            edge_labels[(key1, key2)] = ""
    # if graph is circular use the circular layout
    pos = dict()
    if draw_type == 'circle':
        # get circular positions around (0,0) starting with the root node at (-400,0)
        pos[root_node] = (400, 0)
        angle = 2 * np.pi / (graph.number_of_nodes())
        # iterate over the neighbors of the root node
        cur_node = root_node
        last_node = None
        counter = 0
        while len(pos) < graph.number_of_nodes():
            neighbors = list(graph.neighbors(cur_node))
            for next_node in neighbors:
                if next_node != last_node:
                    counter += 1
                    pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                    last_node = cur_node
                    cur_node = next_node
                    break
    elif draw_type == 'kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.nx_pydot.graphviz_layout(graph)
    # keys to ints
    pos = {int(k): v for k, v in pos.items()}
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_color, width=edge_width)
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax, font_size=8, font_color='black')
    # get node colors from the node labels using the plasma colormap
    cmap = plt.get_cmap('tab20')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=graph_data.node_labels['primary'].num_unique_node_labels)
    node_colors = [cmap(norm(graph_data.node_labels['primary'].node_labels[graph_id][node])) for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors, node_size=node_size)

def snowflakeplot():
    db_name = 'Snowflakes'
    path = '../../../Testing/BenchmarkGraphs/'
    graph_data = get_graph_data(db_name=db_name, data_path=path)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    graph_ids = [0, 100, 200, 300]
    for i in range(4):
        draw_graph(graph_data, graph_ids[i], ax[i], draw_type='kawai')
        ax[i].set_xlabel(f"Graph Label: ${graph_data.graph_labels[graph_ids[i]]}$", fontsize=15)
        # do not the spines
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

    # save the plot as svg
    plt.savefig(f'Plots/{db_name}_plot.pdf', bbox_inches='tight', backend='pgf')
    plt.show()

def m_plot():
    m_0 = M0()
    m_1 = M1()
    m_2 = M1()
    m_3 = M3()

    # get first color of tab20 colormap
    cmap = plt.get_cmap('tab20')
    color = cmap(0)
    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    for i, m in enumerate([m_0, m_1, m_2, m_3]):
        # set positions manually
        pos = {0: (0, 0), 1: (-4, 4), 2: (-2, 4), 3: (2, 4), 4: (4, 4), 5: (-2, 8), 6: (2, 8), 7: (0, 12),
               8: (-4, -4), 9: (-2, -4), 10: (2, -4), 11: (4, -4), 12: (-2, -8), 13: (2, -8), 14: (0, -12)}
        if i == 1:
            #set y to -y
            pos = {k: (x, -y) for k, (x, y) in pos.items()}

        # kawai pos
        #pos = nx.kamada_kawai_layout(m)
        nx.draw_networkx_edges(m, pos=pos, ax=ax[i], edge_color='black', width=0.5)
        nx.draw_networkx_nodes(m, pos=pos, ax=ax[i], node_color=color, node_size=50)
        # draw node ids
        #nx.draw_networkx_labels(m, pos=pos, ax=ax[i], labels={node: node for node in m.nodes()})
        ax[i].set_xlabel(f"$M_{i}$", fontsize=15)
        # do not the box around the plot
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

    # save the plot as svg
    plt.savefig(f'Plots/m_plot.pdf', bbox_inches='tight', backend='pgf')
    plt.show()


def main():
    snowflakeplot()
    m_plot()

if __name__ == '__main__':
    main()