import json
import os
import networkx as nx # Important do not remove
from pathlib import Path

from src.Architectures.ShareGNN.ShareGNNLayers import get_label_string
from src.Preprocessing.create_labels import save_trivial_labels, save_wl_labels, save_primary_labels, \
    save_degree_labels, save_cycle_labels, save_subgraph_labels, save_clique_labels, save_index_labels, \
    save_labeled_degree_labels, save_wl_labeled_labels, save_labels_to_file, save_wl_labeled_edges_labels
from src.Preprocessing.create_properties import write_distance_properties, write_distance_edge_properties
from src.Preprocessing.create_splits import create_splits
from src.TransferLearning.combine_split_files import pretraining_finetuning
from src.Preprocessing.GraphData.GraphData import ShareGNNDataset
from src.utils.GraphLabels import combine_node_labels
from src.Experiment.RunConfiguration import get_run_configs
from src.Preprocessing.load_labels import load_labels
from src.utils.utils import save_graphs


class DatasetPreprocessing:
    """
    Preprocessing class to load the data, generate the splits, labels and properties and save them in the correct folders.
    params:
    db_name: str: name of the dataset
    dataset_configuration: dict: configuration for the dataset
    experiment_configuration: dict: configuration for the experiment
    with_splits: bool: generate the splits
    with_labels_and_properties: bool: generate the labels and properties
    data_generation: str: name of the data generation function
    """
    def __init__(self, dataset_configurations, with_labels_and_properties=True):
        for configuration in dataset_configurations:
            self.db_name = configuration['name']
            self.graph_data = None
            # load the config file
            self.experiment_configuration = configuration
            self.generation_times_labels_path = None
            self.generation_times_properties_path = None

            # create the folders and files for the results and preprocessing data
            self.create_folders_and_files()
            # generate the data only if it does not exist (i.e. the processed folder is empty)
            if not Path(self.experiment_configuration['paths']['data']).joinpath(f'{self.db_name}').joinpath('processed').joinpath(f'data.pt').is_file():
                dataset = self.db_name
                data_generation = self.experiment_configuration['data_generation']
                data_generation_args = self.experiment_configuration.get('generate_function_args', None)
                self.generate_data(dataset, data_generation, data_generation_args)
            self.load_data()
            # generate the split files
            self.generate_configuration_splits()

            # generate the labels and properties automatically from the config file
            if with_labels_and_properties:
                self.preprocessing_from_config()


    def create_folders_and_files(self):
        # create config folders if they do not exist
        self.experiment_configuration['paths']['data'].mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['labels'].mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['properties'].mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['splits'].mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['results'].mkdir(exist_ok=True, parents=True)
        # create folders plots, weights, models and results in the results folder under the db_name
        self.experiment_configuration['paths']['results'].joinpath(self.db_name).joinpath('Plots').mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['results'].joinpath(self.db_name).joinpath('Weights').mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['results'].joinpath(self.db_name).joinpath('Models').mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['results'].joinpath(self.db_name).joinpath('Results').mkdir(exist_ok=True, parents=True)


        # if not exists create the generation_times_labels.txt and generation_times_properties.txt in the Results folder
        if not Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_labels.txt').exists():
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_labels.txt'), 'w') as f:
                f.write('Generation times for labels\n')
        if not Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt').exists():
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt'), 'w') as f:
                f.write('Generation times for properties\n')
        self.generation_times_labels_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_labels.txt')
        self.generation_times_properties_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_properties.txt')

    def generate_data(self, dataset, data_generation_type, data_generation_args):
        # generate the data
        if isinstance(data_generation_type, list):
            if not isinstance(data_generation_args, list):
                data_generation_args = [data_generation_args] * len(data_generation_type)
            zip_list = list(zip(data_generation_type, data_generation_args, self.experiment_configuration['single_datasets']))
            for data_gen, data_gen_args, d in zip_list:
                self.generate_data(d, data_gen, data_gen_args)
            # merge the generated datasets
            graphs = []
            for data_generation_type, data_generation_args, dataset in zip_list:
                graphs.append(ShareGNNDataset(root=str(self.experiment_configuration['paths']['data']),
                                              name=dataset,
                                              use_node_attr=self.experiment_configuration.get('use_node_attr', False),
                                              use_edge_attr=self.experiment_configuration.get('use_edge_attr', False),
                                              delete_zero_columns=False,
                                              task=self.experiment_configuration.get('task', None)
                                              ))
                # merge the graphs
            ShareGNNDataset(root=str(self.experiment_configuration['paths']['data']),
                            name=self.experiment_configuration['name'],
                            from_existing_data=graphs,
                            use_node_attr=self.experiment_configuration.get('use_node_attr', False),
                            use_edge_attr=self.experiment_configuration.get('use_edge_attr', False),
                            delete_zero_columns=False,
                            task=self.experiment_configuration.get('task', None),
                            )
            return
        path = Path(self.experiment_configuration['paths']['data'])
        if Path(Path(self.experiment_configuration['paths']['data']) / dataset / 'processed').exists() and len(
                list(Path(Path(self.experiment_configuration['paths']['data']) / dataset / 'processed').iterdir())) > 0:
            print(
                f"Dataset {dataset} already exists in {Path(self.experiment_configuration['paths']['data'])} . Skip the data generation.")
            return
        if isinstance(data_generation_type, str):
            if data_generation_type != 'generate_from_function':
                try:

                    # download the dataset
                    # create a tmp folder to store the dataset
                    if not Path('tmp').exists():
                        Path('tmp').mkdir()
                    self.graph_data = ShareGNNDataset(root=str(self.experiment_configuration['paths']['data']),
                                                      name=dataset,
                                                      from_existing_data=data_generation_type,
                                                      )
                    if not os.path.exists(path.joinpath(Path(dataset))):
                        os.makedirs(path.joinpath(Path(dataset)))
                    # create processed and raw folders in path+dataset
                    if not os.path.exists(path.joinpath(Path(dataset + "/processed"))):
                        os.makedirs(path.joinpath(Path(dataset + "/processed")))
                    if not os.path.exists(path.joinpath(Path(dataset + "/raw"))):
                        os.makedirs(path.joinpath(Path(dataset + "/raw")))
                    #tu_to_nel(dataset=dataset, out_path=Path(self.experiment_configuration['paths']['data']))
                except:
                    print(f'Could not generate {dataset} from TUDataset')
            else:
                print(f'Do not know how to handle data from {data_generation_type}. Do you mean "TUDataset"?')
            pass
        else:
            if data_generation_type is not None:
                if data_generation_args is None:
                    data_generation_args = {}
                try:
                    # generate data
                    graphs, labels =  data_generation_type(**data_generation_args, split_path=Path(self.experiment_configuration['paths']['splits']))
                    # save lists of graphs and labels in the correct graph_format NEL -> Nodes, Edges, Labels
                    save_graphs(Path(self.experiment_configuration['paths']['data']), dataset, graphs, labels, with_degree=False, graph_format='NEL')
                    self.graph_data = ShareGNNDataset(root=str(self.experiment_configuration['paths']['data']),
                                                      name=dataset,
                                                      from_existing_data='NEL',
                                                      task=self.experiment_configuration.get('task', 'graph'),
                                                      )
                except:
                    # raise the error that has occurred
                    print(f'Could not generate {dataset} from function {data_generation_type} with arguments {data_generation_args}')

            else:
                try:
                    self.graph_data = ShareGNNDataset(root=str(self.experiment_configuration['paths']['data']),
                                                      name=dataset,
                                                      from_existing_data='NEL',
                                                      task=self.dataset_configuration.get('task', None)
                                                      )
                except:
                    print(f'Could not process the data from {dataset} with the given configuration.')


    def load_data(self):
        # load graph data from pt files if it exists in the processed folder
        if self.graph_data is None and self.experiment_configuration['paths']['data'].joinpath(f'{self.db_name}').joinpath('processed').exists():
            self.graph_data = ShareGNNDataset(root=str(self.experiment_configuration['paths']['data']),
                                              name=self.db_name,
                                              task=self.experiment_configuration.get('task', None),
                                              )
            # raise an error if the graph data is still None
        if self.graph_data is None:
            raise ValueError(f'Could not load the graph data for {self.db_name} from {self.experiment_configuration["paths"]["data"]}. Please check the configuration and the data generation function.')


    def generate_configuration_splits(self):
        splits_path = self.experiment_configuration['paths']['splits']
        if 'pretraining_datasets' in self.experiment_configuration or 'finetuning_datasets' in self.experiment_configuration:
            # check whether the split file exists
            if 'pretraining_datasets' in self.experiment_configuration:
                self.experiment_configuration['split_appendix'] = 'pretraining_' + '_'.join(self.experiment_configuration['pretraining_datasets'])
                split_string = f'{self.db_name}_{self.experiment_configuration["split_appendix"]}_splits.json'
                new_splits_path = splits_path.joinpath(split_string)
                if new_splits_path.exists():
                    return
            elif 'finetuning_datasets' in self.experiment_configuration:
                self.experiment_configuration['split_appendix'] = 'finetuning_' + '_'.join(self.experiment_configuration['finetuning_datasets'])
                split_string = f'{self.db_name}_{self.experiment_configuration["split_appendix"]}_splits.json'
                new_splits_path = splits_path.joinpath(split_string)
                if new_splits_path.exists():
                    return
            # otherwise check whether all split files for the datasets exist
            for dataset in self.experiment_configuration['single_datasets']:
                if not splits_path.joinpath(f'{dataset}_splits.json').exists():
                    self.create_split_file()
            paths = [splits_path for dataset in self.experiment_configuration['single_datasets']]
            datasets = self.experiment_configuration['single_datasets']
            pretraining_ids = []
            finetuning_ids = []
            if 'pretraining_datasets' in self.experiment_configuration:
                # get the ids by positions in the self.experiment_configuration['single_datasets']
                pretraining_ids = [self.experiment_configuration['single_datasets'].index(dataset) for dataset in self.experiment_configuration['pretraining_datasets']]
            if 'finetuning_datasets' in self.experiment_configuration:
                # get the ids by positions in the self.experiment_configuration['single_datasets']
                finetuning_ids = [self.experiment_configuration['single_datasets'].index(dataset) for dataset in self.experiment_configuration['finetuning_datasets']]
            # create the pretraining respective finetuning splits
            pretraining_finetuning(paths, datasets, pretraining_ids=pretraining_ids, finetuning_ids=finetuning_ids)

        else:
            splits_path = splits_path.joinpath(f'{self.db_name}_splits.json')

        if splits_path.exists():
            pass
        else:
            self.create_split_file()

        # copy the splits to the processed folder
        #if not Path(self.experiment_configuration['paths']['data']).joinpath(f'{self.db_name}').joinpath('processed').exists():
        #    Path(self.experiment_configuration['paths']['data']).joinpath(f'{self.db_name}').joinpath('processed').mkdir()
        #if 'split_appendix' in self.experiment_configuration:
        #    split_target_path = Path(self.experiment_configuration['paths']['data']).joinpath(f'{self.db_name}').joinpath('processed').joinpath(f'{self.db_name}_{self.experiment_configuration["split_appendix"]}_splits.json')
        #else:
        #    split_target_path = Path(self.experiment_configuration['paths']['data']).joinpath(f'{self.db_name}').joinpath('processed').joinpath(f'{self.db_name}_splits.json')
        # copy the content of the split file to the target path
        #split_target_path.write_text(splits_path.read_text())

    def create_split_file(self):
        # create the splits
        if self.experiment_configuration.get('with_splits', True):
            create_splits(self.db_name, Path(self.experiment_configuration['paths']['data']),
                          Path(self.experiment_configuration['paths']['splits']),
                          folds=self.experiment_configuration['validation_folds'], graph_data=self.graph_data)
        else:
            if self.experiment_configuration.get('split_function', None) is not None:
                # generate splits
                self.experiment_configuration['split_function'](self.experiment_configuration['paths']['splits'],
                                                                **self.experiment_configuration['split_function_args'],
                                                                graph_data=self.graph_data)
            else:
                raise ValueError(
                    f'Please specify a split function in the main config file for the dataset {self.db_name} using the key "split_function".')

    def layer_to_labels(self, layer_strings: json)->Path:
        file_path = None
        layer = json.loads(layer_strings)
        label_path = self.experiment_configuration['paths']['labels'].joinpath(f'{self.graph_data.name}')
        # check if the path exists, otherwise create it
        if not label_path.exists():
            label_path.mkdir()
        # if label_type is a list, then the layer is a combination of different label types
        if type(layer['label_type']) == list and len(layer['label_type']) > 1:
            # recursively call the function for each label type
            labels = []
            label_names = []
            for label_type in layer['label_type']:
                new_layer_string = layer.copy()
                # remove the label_type key and replace it with the new label_type
                new_layer_string['label_type'] = label_type
                l_path = self.layer_to_labels(json.dumps(new_layer_string))
                # get all after last /
                label_name = '_'.join(l_path.stem.split('_')[1:-1])
                label_names.append(label_name)
                labels.append(load_labels(l_path))
            # combine the labels
            combined_labels = combine_node_labels(labels)
            l = f'{combined_labels.label_name}'
            max_labels = layer.get('max_labels', None)
            if max_labels is not None:
                l += f'_{max_labels}'
            file_path = label_path.joinpath(f"{self.graph_data.name}_labels_{l}.pt")
            save_labels_to_file(file_path, combined_labels.dataset_name, l, combined_labels.node_labels, max_labels=layer.get('max_labels', None))
        else:
            if isinstance(layer['label_type'], list):
                layer['label_type'] = layer['label_type'][0]
            # switch case for the different layers
            if layer['label_type'] == 'primary':
                file_path = save_primary_labels(graph_data=self.graph_data,
                                                label_path=label_path,
                                                max_labels=layer.get('max_labels', None),
                                                save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'trivial':
                file_path = save_trivial_labels(graph_data=self.graph_data,
                                                label_path=label_path,
                                                save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'index':
                file_path = save_index_labels(graph_data=self.graph_data,
                                              max_labels=layer.get('max_labels', None),
                                              label_path=label_path,
                                              save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'index_text':
                file_path = save_index_labels(graph_data=self.graph_data,
                                              max_labels=layer.get('max_labels', None),
                                              label_path=label_path,
                                              save_times=self.generation_times_labels_path,
                                                index_text=True)
            elif layer['label_type'] == 'degree':
                file_path = save_degree_labels(graph_data=self.graph_data,
                                               label_path=label_path,
                                               max_labels=layer.get('max_labels', None),
                                               save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'wl':
                layer['max_labels'] = layer.get('max_labels', None)
                layer['depth'] = layer.get('depth', 3)
                if layer['depth'] == 0:
                    file_path = save_degree_labels(graph_data=self.graph_data,
                                                   label_path=label_path,
                                                   max_labels=layer.get('max_labels', None),
                                                   save_times=self.generation_times_labels_path)
                else:
                    file_path = save_wl_labels(graph_data=self.graph_data,
                                               depth=layer.get('depth', 3),
                                               max_labels=layer['max_labels'],
                                               label_path=label_path,
                                               save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'wl_labeled':
                layer['max_labels'] = layer.get('max_labels', None)
                layer['depth'] = layer.get('depth', 3)
                base_labels = None
                if 'base_labels' in layer:
                    base_labels = dict()
                    base_label_path = self.experiment_configuration['paths']['labels'].joinpath(f'{self.graph_data.name}').joinpath(f"{self.graph_data.name}_labels_{get_label_string(layer['base_labels'])}.pt")
                    base_labels['layer_dict'] = layer['base_labels']
                    base_labels['layer_string'] = get_label_string(layer['base_labels'])
                    base_labels['labels'] = load_labels(base_label_path)

                if layer['depth'] == 0:
                    file_path = save_labeled_degree_labels(graph_data=self.graph_data,
                                                           label_path=label_path,
                                                              max_labels=layer.get('max_labels', None),
                                                           save_times=self.generation_times_labels_path)
                else:
                    file_path = save_wl_labeled_labels(graph_data=self.graph_data,
                                                       depth=layer.get('depth', 3),
                                                       max_labels=layer['max_labels'],
                                                       label_path=label_path,
                                                       base_labels=base_labels,
                                                       save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'wl_labeled_edges':
                layer['max_labels'] = layer.get('max_labels', None)
                layer['depth'] = layer.get('depth', 3)
                base_labels = None
                if 'base_labels' in layer:
                    base_labels = dict()
                    base_label_path = self.experiment_configuration['paths']['labels'].joinpath(f'{self.graph_data.name}').joinpath(f"{self.graph_data.name}_labels_{get_label_string(layer['base_labels'])}.pt")
                    base_labels['layer_dict'] = layer['base_labels']
                    base_labels['layer_string'] = get_label_string(layer['base_labels'])
                    base_labels['labels'] = load_labels(base_label_path)


                file_path = save_wl_labeled_edges_labels(graph_data=self.graph_data,
                                                   depth=layer.get('depth', 3),
                                                   max_labels=layer['max_labels'],
                                                   label_path=label_path,
                                                   base_labels=base_labels,
                                                   save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'simple_cycles' or layer['label_type'] == 'induced_cycles':
                cycle_type = 'simple' if layer['label_type'] == 'simple_cycles' else 'induced'
                if 'max_labels' not in layer:
                    layer['max_labels'] = None
                if 'max_cycle_length' not in layer:
                    layer['max_cycle_length'] = None
                if 'min_cycle_length' not in layer:
                    layer['min_cycle_length'] = None
                file_path = save_cycle_labels(graph_data=self.graph_data,
                                                min_cycle_length=layer['min_cycle_length'],
                                              max_cycle_length=layer['max_cycle_length'],
                                              max_labels=layer["max_labels"],
                                              cycle_type=cycle_type,
                                              label_path=label_path,
                                              save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'subgraph':
                if 'id' in layer:
                    if layer['id'] > len(self.experiment_configuration['subgraphs']):
                        raise ValueError(f'Please specify the subgraphs in the config files under the key "subgraphs" as folllows: subgraphs: - "[nx.complete_graph(4)]"')
                    else:
                        subgraph_list = eval(self.experiment_configuration['subgraphs'][layer['id']])
                        file_path = save_subgraph_labels(graph_data=self.graph_data,
                                                         subgraphs=subgraph_list,
                                                         subgraph_id=layer['id'],
                                                         max_labels=layer.get('max_labels', None),
                                                         label_path=label_path,
                                                         save_times=self.generation_times_labels_path)
                else:
                    raise ValueError(f'Please specify the id of the subgraph in the layer with description {layer_strings}.')
            elif layer['label_type'] == 'cliques':
                if 'max_labels' not in layer:
                    layer['max_labels'] = None
                if 'max_clique_size' not in layer:
                    layer['max_clique_size'] = None
                file_path = save_clique_labels(graph_data=self.graph_data,
                                               max_clique=layer['max_clique_size'],
                                               max_labels=layer.get('max_labels', None),
                                               label_path=label_path,
                                               save_times=self.generation_times_labels_path)
            else:
                # print in red in the console
                print(f'The automatic generation of labels for the layer type {layer["label_type"]} is not supported yet.')
        return file_path

    def property_to_properties(self, property_strings: json):
        properties_path = self.experiment_configuration['paths']['properties'].joinpath(f'{self.graph_data.name}')
        # check if the path exists, otherwise create it
        if not properties_path.exists():
            properties_path.mkdir()
        # switch case for the different properties
        properties = json.loads(property_strings)
        if properties['name'] == 'distances':
            if 'cutoff' not in properties:
                properties['cutoff'] = None
            print(f'Generating distance properties for {self.graph_data.name} with cutoff {properties["cutoff"]}')
            write_distance_properties(self.graph_data, out_path=properties_path, cutoff=properties['cutoff'],  save_times=self.generation_times_properties_path)
        # TODO: change the edge_label_distances to the new torch format
        elif properties['name'] == 'edge_label_distances':
            if 'cutoff' not in properties:
                properties['cutoff'] = None
            print(f'Generating edge label distance properties for {self.graph_data.name} with cutoff {properties["cutoff"]}')
            write_distance_edge_properties(self.graph_data, out_path=properties_path, cutoff=properties['cutoff'],  save_times=self.generation_times_properties_path)

    # generate preprocessing by scanning the config file
    def preprocessing_from_config(self):
        # get the layers from the config file
        run_configs = get_run_configs(self.experiment_configuration)
        # preprocessed layers
        preprocessed_label_dicts = set()
        proprocessed_label_dicts_first = set()
        preprocessed_properties = set()
        # iterate over the layers
        for run_config in run_configs:
            for layer in run_config.layers:
                for property_dict in layer.get_unique_property_dicts():
                    p_dict = property_dict.property_dict.copy()
                    p_dict.pop('values')
                    json_property = json.dumps(p_dict, sort_keys=True)
                    preprocessed_properties.add(json_property)
                for label_dict in layer.get_unique_layer_dicts():
                    json_layer = json.dumps(label_dict, sort_keys=True)
                    preprocessed_label_dicts.add(json_layer)
                    # if key base_labels in label_dict, then add the base_labels to the preprocessed_label_dicts
                    if 'base_labels' in label_dict:
                        json_layer = json.dumps(label_dict['base_labels'], sort_keys=True)
                        proprocessed_label_dicts_first.add(json_layer)
        # generate all necessary labels and properties, first need to create the nx graphs to run the algorithms on
        #self.graph_data.create_nx_graphs(directed=False)
        for layer in proprocessed_label_dicts_first:
            self.layer_to_labels(layer)
        for layer in preprocessed_label_dicts:
            self.layer_to_labels(layer)
        for preprocessed_property in preprocessed_properties:
            self.property_to_properties(preprocessed_property)





