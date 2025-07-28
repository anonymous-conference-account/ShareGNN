import ast
import gzip
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

from src.utils.utils import convert_to_tuple


class NodeLabels:
    def __init__(self, dataset_name:str, label_name:str, node_labels:torch.Tensor):
        # first column are the original node labels, the second column are the relabeled node labels
        self.dataset_name = dataset_name
        self.label_name = label_name
        self.original_node_labels = node_labels[:, 0]
        self.node_labels = node_labels[:, 1]
        self.unique_node_labels, self.unique_node_labels_count = torch.unique(self.node_labels, return_counts=True)
        self.num_unique_node_labels = len(self.unique_node_labels)

    def __iadd__(self, other):
        pass



def combine_node_labels(labels: List[NodeLabels]):
    graph_name = labels[0].dataset_name
    combined_label_name = '_'.join([l.label_name for l in labels])
    # stack all the node labels
    stacked_labels = torch.stack([l.node_labels for l in labels], dim=1)
    # get all indices of the unique labels tensors containing -1
    first_entry = stacked_labels[:, 0] == -1
    second_entry = stacked_labels[:, 1] == -1
    # compute or between the two tensors
    or_entries = torch.logical_or(first_entry, second_entry)
    # get the indices of True values
    invalid_indices = torch.where(or_entries)[0]
    # set stack_labels to max, max
    max_first = torch.max(stacked_labels[:, 0] + 1)
    max_second = torch.max(stacked_labels[:, 1] + 1)
    stacked_labels[invalid_indices] = torch.tensor([max_first, max_second])
    # get all unique rows
    unique_labels, new_labels = torch.unique(stacked_labels, return_inverse=True, dim=0)

    artificial_label = len(unique_labels) - 1
    # get frequency of each value in the new labels
    unique_labels_count = torch.bincount(new_labels)
    # set count for invalid indices to 0
    unique_labels_count[artificial_label] = 0
    # sort the unique labels by the frequency and keep the indices
    sorted_indices = torch.argsort(unique_labels_count, descending=True, stable=True)
    # reindex the unique labels: most frequent label is 0, second most frequent is 1, ...
    frequency_sorted_labels = new_labels.new(sorted_indices).argsort()[new_labels]
    new_labels[invalid_indices] = -1
    frequency_sorted_labels[invalid_indices] = -1
    return NodeLabels(graph_name, combined_label_name, torch.stack([new_labels, frequency_sorted_labels], dim=1))

class EdgeLabels:
    def __init__(self):
        self.edge_labels = None
        self.unique_edge_labels = None
        self.db_unique_edge_labels = None
        self.num_unique_edge_labels = 0


class Properties:
    def __init__(self, path: Path, db_name: str, property_name: str, valid_values: dict[tuple[int, int], list[int]]):
        self.name = property_name
        self.db = db_name
        self.valid_values = {}
        self.all_values = None
        # load the properties from a file, first decompress the file with gzip and then load the pickle file
        self.properties = None
        self.properties_slices = None
        self.num_properties = {}
        self.valid_property_map = {}

        # path to the data
        data_path = path.joinpath(db_name).joinpath(f'{db_name}_properties_{property_name}.pt')
        # path to the info file
        info_path = path.joinpath(db_name).joinpath(f'{db_name}_properties_{property_name}.yml')

        # check if the file exists, otherwise raise an error
        if os.path.isfile(data_path) and os.path.isfile(info_path):
            with gzip.open(data_path, 'rb') as f:
                self.all_values, self.properties, self.properties_slices = pickle.load(f)
        else:
            raise FileNotFoundError(f'File {data_path} or {info_path} not found')

        for (layer_id, channel_id), values in valid_values.items():
            self.add_properties(layer_id=layer_id, channel_id=channel_id, valid_values=values)


    def add_properties(self, valid_values: List[int], layer_id: int, channel_id: int):
        self.valid_values[(layer_id, channel_id)] = []
        self.valid_property_map[(layer_id, channel_id)] = {}
        # if property name is edge_label_distance, and the valid values is a list of values interpret them as the distances and take all the values from self.all_values with first entry equal to the distance
        if 'edge_label_distances' in self.name:
            # check if valid_values is a list of ints
            if type(valid_values[0]) == int:
                tmp_valid_values = []
                for v in self.all_values:
                    if v[0] in valid_values:
                        tmp_valid_values.append(v)
                self.valid_values[(layer_id, channel_id)] = tmp_valid_values
            else:
                self.valid_values[(layer_id, channel_id)] = valid_values
        elif 'circle_distances' in self.name:
            if type(valid_values[0]) == str:
                for v in valid_values:
                    if v == 'no_circles':
                        for x in self.all_values:
                            if x[1] == 0 and x[2] == 0:
                                self.valid_values[(layer_id, channel_id)].append(x)
                    if v == 'circles':
                        for x in self.all_values:
                            if x[1] == 1 and x[2] == 1:
                                self.valid_values[(layer_id, channel_id)].append(x)
                    if v == 'in_circles':
                        for x in self.all_values:
                            if x[1] == 0 and x[2] == 1:
                                self.valid_values[(layer_id, channel_id)].append(x)
                    if v == 'out_circles':
                        for x in self.all_values:
                            if x[1] == 1 and x[2] == 0:
                                self.valid_values[(layer_id, channel_id)].append(x)
            else:
                self.valid_values[(layer_id, channel_id)] = valid_values
        else:
            self.valid_values[(layer_id, channel_id)] = valid_values

        # check if all the valid values are in the valid properties, if not raise an error
        invalid_values = []
        for value in self.valid_values[(layer_id, channel_id)]:
            if value not in self.all_values:
                invalid_values.append(value)
        if len(invalid_values) > 0:
            # remove invalid values from the valid values
            self.valid_values[(layer_id, channel_id)] = [v for v in self.valid_values[(layer_id, channel_id)] if v not in invalid_values]
            print(f'There are properties that are not arising in the dataset: {invalid_values}')

        # number of valid properties
        self.num_properties[(layer_id, channel_id)] = len(self.valid_values[(layer_id, channel_id)])
        for i, value in enumerate(self.valid_values[(layer_id, channel_id)]):
            try:
                property_value = int(value)
                self.valid_property_map[(layer_id, channel_id)][property_value] = i
            except:
                # check if the length of the value is 1, if not iterate over the values
                try:
                    len(value[0])
                    for v in value:
                        self.valid_property_map[(layer_id, channel_id)][convert_to_tuple(v)] = i
                except:
                    self.valid_property_map[(layer_id, channel_id)][convert_to_tuple(value)] = i

