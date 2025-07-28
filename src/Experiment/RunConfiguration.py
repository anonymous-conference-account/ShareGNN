import itertools
from typing import List

import numpy as np
from sklearn.utils.extmath import cartesian
from torch import cartesian_prod

from src.Architectures.ShareGNN.ShareGNNLayers import Layer
class RunConfiguration:
    def __init__(self, config, network_architecture, layers, batch_size, lr, epochs, dropout, optimizer, weight_decay, loss, task="classification"):
        self.config = config
        self.network_architecture = network_architecture.copy()
        self.layers = layers.copy()
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.loss = loss
        self.task = task

    def print(self):
        print(f"Network architecture: {self.network_architecture}")
        print(f"Layers: {self.layers}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Epochs: {self.epochs}")
        print(f"Dropout: {self.dropout}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Loss: {self.loss}")

def generate_layer_options(layer_dict):
    options = []
    for label_type in layer_dict['labels']:
        properties_dict = []
        base_dict = {}
        for key, value in layer_dict.items():
            if key != 'labels' and key != 'properties':
                base_dict[key] = value
        if layer_dict.get('properties', None) is not None:
            for prop_val in layer_dict['properties']:
                properties_dict.append(prop_val)
        key_list = list(label_type.keys())
        # remove label_type from key list
        key_list.remove('label_type')
        # remove all keys where the value is None or an empty list
        for key in key_list:
            if label_type[key] is None or (type(label_type[key]) == list and len(label_type[key]) == 0):
                key_list.remove(key)
        if len(key_list) == 0:
            if properties_dict is None or len(properties_dict) == 0:
                options.append({'layer_type': layer_dict['layer_type'], 'channels': [{'labels': label_type}]})
            else:
                for prop_val in properties_dict:
                    options.append({'layer_type': layer_dict['layer_type'], 'channels': [{'labels': label_type, 'properties': prop_val.copy()}]})
        else:
            value_list = []
            for key in key_list:
                if type(label_type[key]) != list:
                    value_list.append([label_type[key]])
                else:
                    value_list.append(label_type[key])
            # get all value combinations as tuples over the value lists
            value_combinations = []
            for i in range(len(value_list)):
                if len(value_combinations) == 0:
                    for v in value_list[i]:
                        value_combinations.append([v])
                else:
                    new_combinations = []
                    for c in value_combinations:
                        for v in value_list[i]:
                            new_combinations.append(c + [v])
                    value_combinations = new_combinations
            for i, values in enumerate(value_combinations):
                if properties_dict is None or len(properties_dict) == 0:
                    curr_layer_dict = base_dict
                    channels_list = []
                    label_dict = {'label_type': label_type['label_type']}
                    for j, value in enumerate(values):
                        label_dict[key_list[j]] = value
                    channels_list.append({'labels' : label_dict})
                    curr_layer_dict['channels'] = channels_list
                    options.append(curr_layer_dict)
                else:
                    for prop_val in properties_dict:
                        curr_layer_dict = base_dict
                        channels_list = []
                        label_dict = {'label_type': label_type['label_type']}
                        for j, value in enumerate(values):
                            label_dict[key_list[j]] = value
                        channels_list.append({'labels' : label_dict, 'properties': prop_val.copy()})
                        curr_layer_dict['channels'] = channels_list
                        options.append(curr_layer_dict)
    return options


def preprocess_network_architectures(network_architectures_dict):
    network_architectures = []
    for network_id, network_architecture in enumerate(network_architectures_dict):
        current_network_architectures = []
        layers_per_architecture = []
        for i, layer in enumerate(network_architecture):
            correct, error = check_layer(i, layer)
            if correct:
                if len(layers_per_architecture) <= i:
                    while len(layers_per_architecture) <= i:
                        layers_per_architecture.append([])
                layers_per_architecture[i].append(network_architecture[i])
            else:
                short_type, error = check_layer_short_type(i, layer)
                if short_type:
                    if isinstance(layer['heads'], int):
                        channel_combinations = [layer['heads']]
                    elif isinstance(layer['heads'], list):
                        channel_combinations = layer['heads']
                    else:
                        raise ValueError(f'Channels in layer {i} is not an int or a list')
                    label_combinations = len(layer['labels'])
                    key_combinations_per_label = []
                    key_list_per_label = [None] * label_combinations
                    for j, label in enumerate(layer['labels']):
                        # get all keys except label_type
                        key_list = list(label.keys())
                        key_list.remove('label_type')
                        key_list_per_label[j] = key_list
                        # get all lenghts of the values of the keys
                    for j, key_list in enumerate(key_list_per_label):
                        if len(key_list) == 0:
                            key_combinations_per_label.append([1])
                        else:
                            key_combinations_per_label.append([])
                            for key in key_list:
                                key_combinations_per_label[-1].append(len(layer['labels'][j][key]))
                    if layer['layer_type'] == 'convolution':
                        property_combinations = len(layer['properties'])
                    else:
                        property_combinations = 0

                    # get all possible combinations of the values in triples
                    option_dicts = []
                    for num_channels in channel_combinations:
                        for label_id in range(label_combinations):
                            num_different_keys = len(key_list_per_label[label_id])
                            # generate all choices from a list of counts
                            all_choices = []
                            for j in range(num_different_keys):
                                if len(all_choices) == 0:
                                    for k in range(key_combinations_per_label[label_id][j]):
                                        all_choices.append([k])
                                else:
                                    new_choices = []
                                    for choice in all_choices:
                                        for k in range(key_combinations_per_label[label_id][j]):
                                            new_choices.append(choice + [k])
                                    all_choices = new_choices
                            for key_combinations in range(np.prod(key_combinations_per_label[label_id])):
                                if layer['layer_type'] == 'convolution':
                                    for property_id in range(property_combinations):
                                        option_dict = {}
                                        option_dict['layer_type'] = layer['layer_type']
                                        option_dict['activation'] = layer.get('activation', None)
                                        option_dict['activation_kwargs'] = layer.get('activation_kwargs', {})
                                        option_dict['heads'] = []
                                        option_dict['concatenate_heads'] = layer.get('concatenate_heads', True)
                                        for j in range(num_channels):
                                            channel_dict = {}
                                            channel_dict['bias'] = layer['bias']
                                            channel_dict['activation'] = layer.get('activation', None)
                                            channel_dict['activation_kwargs'] = layer.get('activation_kwargs', {})
                                            label_dict = {}
                                            label_dict['head'] = {'label_type': layer['labels'][label_id]['label_type']}
                                            label_dict['tail'] = {'label_type': layer['labels'][label_id]['label_type']}
                                            label_dict['bias'] = {'label_type': layer['labels'][label_id]['label_type']}
                                            if len(all_choices) > 0:
                                                choice = all_choices[key_combinations]
                                                for c_idx, value in enumerate(choice):
                                                    key = key_list_per_label[label_id][c_idx]
                                                    label_dict['head'][key] = layer['labels'][label_id][key][value]
                                                    label_dict['tail'][key] = layer['labels'][label_id][key][value]
                                                    label_dict['bias'][key] = layer['labels'][label_id][key][value]
                                            channel_dict['labels'] = label_dict.copy()
                                            channel_dict['properties'] = {}
                                            channel_dict['properties']['name'] = layer['properties'][property_id]['name']
                                            channel_dict['properties']['values'] = layer['properties'][property_id]['values']
                                            if layer['properties'][property_id].get('cutoff', None) is not None:
                                                channel_dict['properties']['cutoff'] = layer['properties'][property_id]['cutoff']
                                            option_dict['heads'].append(channel_dict)
                                        option_dicts.append(option_dict)
                                else:
                                    option_dict = {}
                                    option_dict['layer_type'] = layer['layer_type']
                                    option_dict['activation'] = layer.get('activation', None)
                                    option_dict['activation_kwargs'] = layer.get('activation_kwargs', {})
                                    if layer.get('out_dim', None) is not None:
                                        option_dict['out_dim'] = layer['out_dim']
                                    option_dict['heads'] = []
                                    option_dict['concatenate_heads'] = layer.get('concatenate_heads', True)
                                    for j in range(num_channels):
                                        channel_dict = {}
                                        channel_dict['bias'] = layer['bias']
                                        channel_dict['activation'] = layer.get('activation', None)
                                        channel_dict['activation_kwargs'] = layer.get('activation_kwargs', {})
                                        label_dict = {'label_type': layer['labels'][label_id]['label_type']}
                                        if len(all_choices) > 0:
                                            choice = all_choices[key_combinations]
                                            for c_idx, value in enumerate(choice):
                                                key = key_list_per_label[label_id][c_idx]
                                                label_dict[key] = layer['labels'][label_id][key][value]
                                        channel_dict['labels'] = label_dict.copy()
                                        option_dict['heads'].append(channel_dict)
                                    option_dicts.append(option_dict)

                    if len(layers_per_architecture) <= i:
                        while len(layers_per_architecture) <= i:
                            layers_per_architecture.append([])
                    layers_per_architecture[i] = option_dicts.copy()
                else:
                    raise ValueError(f'Layer {i} is not correctly defined: {error}')
        # cartesian product over all entries of layers_per_architecture
        combinations = [x for x in itertools.product(*layers_per_architecture)]

        network_architectures += [list(x) for x in combinations]
    return network_architectures


def check_layer(i:int, layer: dict)->(bool, str):
    if 'layer_type' not in layer:
        return False, f'Layer type not defined in layer {i}, it must be convolution or aggregation'
    if layer['layer_type'] == 'linear':
        pass
    elif layer['layer_type'] == 'reshape':
        pass
    elif layer['layer_type'] == 'layer_norm':
        pass
    else:
        if 'heads' not in layer:
            return False, f'Channels not defined in layer {i}'
        else:
            if not isinstance(layer['heads'], list):
                return False, f'Channels must be a list in layer {i}'
            else:
                for channel in layer['heads']:
                    if not isinstance(channel, dict):
                        return False, f'Channel must be a dictionary in layer {i}'
                    if 'bias' not in channel:
                        return False, f'Bias not defined in layer {i}, it must be True or False'
                    if 'labels' not in channel:
                        return False, f'Labels not defined in channel {i}'
                    else:
                        if layer['layer_type'] == 'convolution':
                            if 'head' not in channel['labels']:
                                return False, f'Head not defined in channel {i}'
                            else:
                                if 'label_type' not in channel['labels']['head']:
                                    return False, f'Label type not defined in channel {i} for head'
                            if 'tail' not in channel['labels']:
                                return False, f'Tail not defined in channel {i}'
                            else:
                                if 'label_type' not in channel['labels']['tail']:
                                    return False, f'Label type not defined in channel {i} for tail'
                            if 'bias' not in channel['labels']:
                                return False, f'Bias not defined in channel {i}'
                            else:
                                if 'label_type' not in channel['labels']['bias']:
                                    return False, f'Label type not defined in channel {i} for bias'
                            if 'properties' not in channel:
                                return False, f'Properties not defined in channel {i}'
                            else:
                                if 'name' not in channel['properties']:
                                    return False, f'Property name not defined in channel {i}'
                                if 'values' not in channel['properties']:
                                    return False, f'Property values not defined in channel {i}'
                                else:
                                    if not isinstance(channel['properties']['values'], list):
                                        return False, f'Property values must be a list in channel {i}'
                        elif layer['layer_type'] == 'aggregation':
                            if 'label_type' not in channel['labels']:
                                return False, f'Label type not defined in channel {i}'

    return True, ''

def check_layer_short_type(i, layer):
    if 'layer_type' not in layer:
        return False, f'Layer type not defined in layer {i}, it must be convolution or aggregation'
    if 'bias' not in layer:
        return False, f'Bias not defined in layer {i}, it must be True or False'
    if 'heads' not in layer:
        return False, f'Channels not defined in layer {i}, it must be a list of ints of parallel channels used'
    else:
        if not isinstance(layer['heads'], list):
            return False, f'Channels must be a list in layer {i}'
    if 'labels' not in layer:
        return False, f'Labels not defined in layer {i}'
    else:
        if not isinstance(layer['labels'], list):
            return False, f'Labels must be a list in layer {i}'
        else:
            for l in layer['labels']:
                if not isinstance(l, dict):
                    return False, f'Label must be a dictionary in layer {i}'
                if 'label_type' not in l:
                    return False, f'Label type not defined in label in layer {i}'

    if 'properties' not in layer:
        if layer['layer_type'] == 'convolution':
            return False, f'Properties not defined in convolution layer {i}'
    else:
        if not isinstance(layer['properties'], list):
            return False, f'Properties must be a list in layer {i}'
        else:
            for prop in layer['properties']:
                if not isinstance(prop, dict):
                    return False, f'Property must be a dictionary in layer {i}'
                if 'name' not in prop:
                    return False, f'Property name not defined in layer {i}'
                if 'values' not in prop:
                    return False, f'Property values not defined in layer {i}'
                else:
                    if not isinstance(prop['values'], list):
                        return False, f'Property values must be a list'
    return True, ''

def check_network_architectures(network_architectures, print_errors=False):
    '''
    Checks if the network architectures is in the correct format
    '''
    invalid_architectures = []
    for i, architecture in enumerate(network_architectures):
        invalid_layers = []
        for i, layer in enumerate(architecture):
            correct, error = check_layer(i, layer)
            if not correct:
                invalid_layers.append(error)
        if len(invalid_layers) > 0:
            invalid_architectures.append(i)
            if print_errors:
                for error in invalid_layers:
                        print(error)
    if len(invalid_architectures) > 0:
        if print_errors:
            for i in invalid_architectures:
                print(f'Architecture {i} is invalid')
        return False
    return True



def get_run_configs(experiment_configuration):
    # define the network type from the config file
    run_configs = []
    task = "graph_classification" #default task is graph classification
    if 'task' in experiment_configuration:
        task = experiment_configuration['task']
    # get networks from the config file and preprocess them
    # bring the config file network architecture into the correct format
    if not experiment_configuration.get('model', 'ShareGNN') == 'ShareGNN':
        network_architectures = preprocess_network_architectures_ordinary(experiment_configuration)
    else:
        network_architectures = preprocess_network_architectures(experiment_configuration['networks'])
        if not check_network_architectures(network_architectures, print_errors=True):
            raise ValueError('Network architecture not correctly defined')
    # iterate over all network architectures
    for network_architecture in network_architectures:
        layers = []
        # get all different run configurations
        for i, l in enumerate(network_architecture):
            layers.append(Layer(l, i))
        for b in experiment_configuration.get('batch_size', [128]):
            for lr in experiment_configuration.get('learning_rate', [0.001]):
                for e in experiment_configuration.get('epochs', [100]):
                    for d in experiment_configuration.get('dropout', [0.0]):
                        for o in experiment_configuration.get('optimizer', ['Adam']):
                            for w in experiment_configuration.get('weight_decay', [0.0]):
                                for loss in experiment_configuration.get('loss', ['CrossEntropyLoss']):
                                    run_configs.append(
                                        RunConfiguration(experiment_configuration, network_architecture, layers, b, lr, e, d, o, w, loss, task))
    return run_configs
