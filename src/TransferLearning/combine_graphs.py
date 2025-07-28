import os
from pathlib import Path

from src.Preprocessing.create_labels import  save_cycle_labels, save_subgraph_labels, \
    save_clique_labels
from src.Preprocessing.create_properties import write_distance_properties
from src.Preprocessing.create_splits import create_splits, create_transfer_splits
from src.Preprocessing.GraphData.GraphData import get_graph_data
from src.utils.utils import save_graphs
import networkx as nx


def create_dataset(dataset_name, layers=None, with_degree=False):
    # load the graphs
    data_path = '../GraphData/DS_all/'
    graph_data = get_graph_data(dataset_name, data_path, use_labels=True, use_attributes=False, relabel_nodes=False)
    output_path = 'Data/NEL_Format/'

    # if there exist graph labels -1, 1 shift to 0,1
    if min(graph_data.graph_labels) == -1 and max(graph_data.graph_labels) == 1:
        for i, label in enumerate(graph_data.graph_labels):
            graph_data.graph_labels[i] += 1
            graph_data.graph_labels[i] //= 2
    # if the graph labels start from 1, shift to 0,1,2 ...
    if min(graph_data.graph_labels) == 1:
        for i, label in enumerate(graph_data.graph_labels):
            graph_data.graph_labels[i] -= 1


    save_graphs(path=output_path, db_name=f'{dataset_name}', graphs=graph_data.graphs, labels=graph_data.graph_labels, with_degree=with_degree, graph_format='NEL')


def combine_nel_graphs(data_path:Path, output_path:Path, dataset_names):
    edge_files = []
    label_files = []
    node_files = []
    new_name = ''
    num_graphs_per_dataset = []

    for i, dataset in enumerate(dataset_names):
        if i == 0:
            new_name = dataset
        else:
            new_name += f'_{dataset}'
        if not Path(data_path.joinpath(f'{dataset}')).exists():
            raise FileNotFoundError(data_path.joinpath(f'{dataset}'))
        edge_files.append(data_path.joinpath(f'{dataset}/raw/{dataset}_Edges.txt'))
        label_files.append(data_path.joinpath(f'{dataset}/raw/{dataset}_Labels.txt'))
        node_files.append(data_path.joinpath(f'{dataset}/raw/{dataset}_Nodes.txt'))
        # get number of graphs in each dataset using the number of lines in the label file
        with open(data_path.joinpath(f'{dataset}/raw/{dataset}_Labels.txt'), 'r') as f:
            num_graphs_per_dataset.append(len(f.readlines()))

    # ceate new folder
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    else:
        Path(output_path.joinpath(f'{new_name}')).mkdir(parents=True, exist_ok=True)
        Path(output_path.joinpath(f'{new_name}/raw')).mkdir(parents=True, exist_ok=True)
        Path(output_path.joinpath(f'{new_name}/processed')).mkdir(parents=True, exist_ok=True)



    # ceate new files from new_name.Eges.txt using edge_files
    with open(output_path.joinpath(f'{new_name}/raw/{new_name}_Edges.txt'), 'w') as f:
        start_index = 0
        for i, edge_file in enumerate(edge_files):
            if i != 0:
                f.write('\n')
            with open(edge_file, 'r') as e:
                # add all the lines but add start_index to the first number in each line
                for line in e.readlines():
                    line = line.split(' ')
                    line[0] = str(int(line[0]) + start_index)
                    f.write(' '.join(line))
            # update the start index for the next dataset
            start_index += num_graphs_per_dataset[i]


    # ceate new files from new_name.Nodes.txt using node_files
    with open(output_path.joinpath(f'{new_name}/raw/{new_name}_Nodes.txt'), 'w') as f:
        start_index = 0
        for i, node_file in enumerate(node_files):
            if i != 0:
                f.write('\n')
            with open(node_file, 'r') as e:
                # add all the lines but add start_index to the first number in each line
                for line in e.readlines():
                    line = line.split(' ')
                    line[0] = str(int(line[0]) + start_index)
                    f.write(' '.join(line))
                # update the start index for the next dataset
            start_index += num_graphs_per_dataset[i]

    # ceate new files from new_name.Labels.txt using label_files
    with open(output_path.joinpath(f'{new_name}/raw/{new_name}_Labels.txt'), 'w') as f:
        start_index = 0
        for i, label_file in enumerate(label_files):
            if i != 0:
                f.write('\n')
            with open(label_file, 'r') as e:
                # add all the lines but add start_index to the first number in each line
                for line in e.readlines():
                    line = line.split(' ')
                    line[1] = str(int(line[1]) + start_index)
                    f.write(' '.join(line))
                # update the start index for the next dataset
            start_index += num_graphs_per_dataset[i]

def transfer_IMDB():
    create_dataset('IMDB-BINARY')
    create_dataset('IMDB-MULTI')
    combine_nel_graphs(['IMDB-BINARY', 'IMDB-MULTI'])
    write_distance_properties(data_path="Data/NEL_Format/", db_name='IMDB-BINARY_IMDB-MULTI', cutoff=2,
                              out_path="Data/Properties/", data_format='NEL')
    save_standard_labels(data_path="Data/NEL_Format/", db_names=['IMDB-BINARY_IMDB-MULTI'], label_path="Data/Labels/", format='NEL')
    for cycle_type in ['induced']:
        for cycle_length in [3, 4, 5, 6, 10, 20, 50]:
            save_cycle_labels(data_path="Data/NEL_Format/", db_names=['IMDB-BINARY_IMDB-MULTI'], cycle_type=cycle_type,
                              max_cycle_length=cycle_length, label_path="Data/Labels/", format='NEL')
    for clique_size in [3, 4, 5, 6, 10, 20, 50]:
        save_clique_labels(data_path="Data/NEL_Format/", db_names=['IMDB-BINARY_IMDB-MULTI'], max_clique=clique_size,
                           label_path="Data/Labels/", format='NEL')
    # create subgraph labels
    save_subgraph_labels(data_path="Data/NEL_Format/", db_names=['IMDB-BINARY_IMDB-MULTI'],
                         subgraphs=[nx.cycle_graph(3), nx.star_graph(1)], label_path="Data/Labels/", id=1, format='NEL')
    create_splits(db_name='IMDB-BINARY_IMDB-MULTI', data_path="Data/NEL_Format/", output_path="Data/Splits/", graph_format='NEL')
    create_transfer_splits(db_name='IMDB-BINARY_IMDB-MULTI',
                           path="Data/NEL_Format/",
                           output_path="Data/Splits/",
                           data_format='NEL',
                           split_type='transfer')
    create_transfer_splits(db_name='IMDB-BINARY_IMDB-MULTI',
                           path="Data/NEL_Format/",
                           output_path="Data/Splits/",
                           data_format='NEL',
                           split_type='mixed')

def transfer_PTC():
    names = ['PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR']
#    for name in names:
#        create_dataset(name)
#    combine_nel_graphs(names)
    combined_name = '_'.join(names)
#    write_distance_properties(data_path="Data/NEL_Format/", db_name=combined_name, cutoff=None,
#                              out_path="Data/Properties/", format='NEL')
#    save_standard_labels(data_path="Data/NEL_Format/", db_names=[combined_name], label_path="Data/Labels/", format='NEL')
#    for cycle_type in ['induced']:
#        for cycle_length in [3, 4, 5, 6, 10, 20, 50]:
#            save_circle_labels(data_path="Data/NEL_Format/", db_names=[combined_name], cycle_type=cycle_type,
#                               length_bound=cycle_length, label_path="Data/Labels/", format='NEL')
#    for clique_size in [3, 4, 5, 6, 10, 20, 50]:
#        save_clique_labels(data_path="Data/NEL_Format/", db_names=[combined_name], max_clique=clique_size,
#                           label_path="Data/Labels/", format='NEL')
#    # create subgraph labels
#    save_subgraph_labels(data_path="Data/NEL_Format/", db_names=[combined_name],
#                         subgraphs=[nx.cycle_graph(3), nx.star_graph(1)], label_path="Data/Labels/", id=1, format='NEL')
#    create_splits(db_name=combined_name, path="Data/NEL_Format/", output_path="Data/Splits/", format='NEL')
#    create_transfer_splits(db_name=combined_name, path="Data/NEL_Format/", output_path="Data/Splits/",
#                           data_format='NEL', split_type='transfer')
    create_transfer_splits(db_name=combined_name, path="Data/NEL_Format/", output_path="Data/Splits/",
                           data_format='NEL', split_type='mixed')

def main():
    # transfer_IMDB()
    transfer_PTC()




if __name__ == "__main__":
    main()
