import os

from src.Preprocessing.GraphData.GraphData import get_graph_data
from src.utils.utils import save_graphs


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


def combine_nel_graphs(dataset_names):
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
        if not os.path.exists(f'Data/NEL_Format/{dataset}'):
            raise FileNotFoundError(f'Data/NEL_Format/{dataset} does not exist')
        edge_files.append(f'Data/NEL_Format/{dataset}/raw/{dataset}_Edges.txt')
        label_files.append(f'Data/NEL_Format/{dataset}/raw/{dataset}_Labels.txt')
        node_files.append(f'Data/NEL_Format/{dataset}/raw/{dataset}_Nodes.txt')
        # get number of graphs in each dataset using the number of lines in the label file
        with open(f'Data/NEL_Format/{dataset}/raw/{dataset}_Labels.txt', 'r') as f:
            num_graphs_per_dataset.append(len(f.readlines()))

    # ceate new folder
    os.makedirs(f'Data/NEL_Format/{new_name}', exist_ok = True)
    os.makedirs(f'Data/NEL_Format/{new_name}/raw', exist_ok = True)
    os.makedirs(f'Data/NEL_Format/{new_name}/processed', exist_ok = True)



    # ceate new files from new_name.Eges.txt using edge_files
    with open(f'Data/NEL_Format/{new_name}/raw/{new_name}_Edges.txt', 'w') as f:
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
    with open(f'Data/NEL_Format/{new_name}/raw/{new_name}_Nodes.txt', 'w') as f:
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
    with open(f'Data/NEL_Format/{new_name}/raw/{new_name}_Labels.txt', 'w') as f:
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



def main():
    #create_dataset('NCI109')
    #create_dataset('NCI1')
    #create_dataset('Mutagenicity')
    #create_dataset('DHFR')
    pass





if __name__ == "__main__":
    main()
