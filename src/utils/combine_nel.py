from pathlib import Path


def combine_nel_graphs(dataset_names, input_dir:Path, output_dir:Path):
    edge_files = []
    label_files = []
    node_files = []
    new_name = ''
    num_graphs_per_dataset = []
    # if output_dir is empty use input_dir
    if not output_dir:
        output_dir = input_dir

    for i, dataset in enumerate(dataset_names):
        if i == 0:
            new_name = dataset
        else:
            new_name += f'_{dataset}'
        if not input_dir.joinpath(dataset).exists():
            raise FileNotFoundError(f'{input_dir.joinpath(dataset)} does not exist')
        edge_files.append(input_dir.joinpath(dataset, 'raw', f'{dataset}_Edges.txt'))
        label_files.append(input_dir.joinpath(dataset, 'raw', f'{dataset}_Labels.txt'))
        node_files.append(input_dir.joinpath(dataset, 'raw', f'{dataset}_Nodes.txt'))
        # get number of graphs in each dataset using the number of lines in the label file
        with open(input_dir.joinpath(dataset, 'raw', f'{dataset}_Labels.txt'), 'r') as f:
            num_graphs_per_dataset.append(len(f.readlines()))

    # ceate new folder in output_dir
    Path(output_dir.joinpath(new_name)).mkdir(parents=True, exist_ok=True)
    Path(output_dir.joinpath(new_name, 'raw')).mkdir(parents=True, exist_ok=True)
    Path(output_dir.joinpath(new_name, 'processed')).mkdir(parents=True, exist_ok=True)



    # ceate new files from new_name.Eges.txt using edge_files
    out_edge_file = output_dir.joinpath(new_name, 'raw', f'{new_name}_Edges.txt')
    with open(out_edge_file, 'w') as f:
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
    out_node_file = output_dir.joinpath(new_name, 'raw', f'{new_name}_Nodes.txt')
    with open(out_node_file, 'w') as f:
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
    out_label_file = output_dir.joinpath(new_name, 'raw', f'{new_name}_Labels.txt')
    with open(out_label_file, 'w') as f:
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
