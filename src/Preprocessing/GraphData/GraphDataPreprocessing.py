# abstract class for graph data preprocessing
import abc
from pathlib import Path

import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import ZINC



class GraphDataPreprocessing(abc.ABC):
    def __init__(self, name, tmp_dir="/tmp"):
        self.name = name
        self.tmp_dir = tmp_dir
        self.processed_dataset = None
        self.slices = None
        self.sizes =  {'num_edge_attributes': None,
                         'num_edge_labels': None,
                         'num_node_attributes': None,
                         'num_node_labels': None
                }


    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        Abstract method to preprocess the raw dataset.
        This method should be implemented by subclasses to perform specific preprocessing tasks.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        return NotImplementedError("Subclasses should implement this method.")


    def set_sizes(self):
        if self.processed_dataset is None:
            raise ValueError("Processed dataset is not set. Please run preprocess() first.")

        self.sizes = {'num_edge_attributes': self.processed_dataset.edge_attributes.shape[-1],
                 'num_edge_labels': len(torch.unique(self.processed_dataset.primary_edge_labels)),
                 'num_node_attributes': self.processed_dataset.node_attributes.shape[-1],
                 'num_node_labels': len(torch.unique(self.processed_dataset.primary_node_labels))
                 }


class ZINCGraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()


    def preprocess(self, *args, **kwargs):
        """
        Preprocess the ZINC dataset.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        subset = True
        if self.name in ['ZINC-full', 'ZINC-Full', 'ZINCFull', 'ZINC-250k']:
            subset = False
        train_data = ZINC(root=self.tmp_dir, subset=subset, split='train')
        validation_data = ZINC(root=self.tmp_dir, subset=subset, split='val')
        test_data = ZINC(root=self.tmp_dir, subset=subset, split='test')
        # merge train_data._data, validation_data._data and test_data._data
        all_data = torch_geometric.data.InMemoryDataset.collate(
            [train_data._data, validation_data._data, test_data._data])

        self.processed_dataset = all_data[0]

        # merge the slices
        self.slices = dict()
        for key in train_data.slices.keys():
            validation_data.slices[key] += train_data.slices[key][-1]
            test_data.slices[key] += validation_data.slices[key][-1]
            self.slices[key] = torch.cat(
                (train_data.slices[key], validation_data.slices[key][1:], test_data.slices[key][1:]))

        self.processed_dataset.primary_node_labels = self.processed_dataset.x
        # flatten
        self.processed_dataset.primary_node_labels = self.processed_dataset.primary_node_labels.view(-1)
        self.slices['primary_node_labels'] = self.slices['x']
        self.processed_dataset.node_attributes = torch.Tensor()
        self.processed_dataset.primary_edge_labels = self.processed_dataset.edge_attr
        self.slices['primary_edge_labels'] = self.slices['edge_attr']
        self.processed_dataset.edge_attributes = torch.Tensor()


        self.set_sizes()

        return self.processed_dataset, self.slices, self.sizes


class QMGraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()


    def preprocess(self, *args, **kwargs):
        """
        Preprocess the QM9 dataset.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        if self.name in ['QM9', 'qm9', 'QM', 'qm']:
            dataset = torch_geometric.datasets.QM9(root=self.tmp_dir)
        elif self.name in ['QM7', 'qm7', 'QM7b', 'qm7b']:
            dataset = torch_geometric.datasets.QM7b(root=self.tmp_dir)
        dataset_node_labels = dataset.data.z
        dataset_node_attributes = dataset.data.x[:, [6, 7, 8, 9]]
        # one hot over the edge_attr
        dataset_edge_labels = torch.argmax(dataset.data.edge_attr, dim=1)

        dataset.data.primary_node_labels = dataset_node_labels
        dataset.data.primary_edge_labels = dataset_edge_labels
        dataset.data.node_attributes = torch.cat((dataset_node_attributes, dataset.pos), dim=1)
        dataset.data.edge_attributes = torch.Tensor()

        self.processed_dataset = dataset.data
        self.slices = dataset.slices
        self.slices['primary_node_labels'] = self.slices['x']
        self.slices['node_attributes'] = self.slices['x']
        self.slices['primary_edge_labels'] = self.slices['edge_attr']
        self.set_sizes()
        return self.processed_dataset, self.slices, self.sizes


class OGBGraphPropertyGraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()


    def preprocess(self, *args, **kwargs):
        """
        Preprocess the OGB dataset.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        dataset_ogb = PygGraphPropPredDataset(name=self.name, root=self.tmp_dir)
        split_idx = dataset_ogb.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        self.processed_dataset = dataset_ogb.data
        self.processed_dataset.primary_node_labels = dataset_ogb.x[:, 0]  # first column is the primary node label
        self.processed_dataset.node_attributes = dataset_ogb.x[:, 1:9]  # next 8 columns are node attributes
        self.processed_dataset.primary_edge_labels = dataset_ogb.edge_attr[:, 0]  # first column is the primary edge label
        self.processed_dataset.edge_attributes = dataset_ogb.edge_attr[:, 1:3]  # next 2 columns are edge attributes

        # if second dimension of y is 1, flatten it
        if self.processed_dataset.y.dim() == 2 and self.processed_dataset.y.shape[1] == 1:
            self.processed_dataset.y = self.processed_dataset.y.view(-1)

        self.slices = dataset_ogb.slices
        self.slices['primary_node_labels'] = self.slices['x']
        self.slices['node_attributes'] = self.slices['x']

        self.slices['primary_edge_labels'] = self.slices['edge_attr']
        self.slices['edge_attributes'] = self.slices['edge_attr']

        self.set_sizes()

        return self.processed_dataset, self.slices, self.sizes

class SubstructureBenchmarkPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()
    def preprocess(self, *args, **kwargs):
        """
        Preprocess the Substructure Benchmark dataset.
        """
        # relative path to project root
        root_path = Path(__file__).parent.parent.parent.parent
        # root_path to string
        root_path = str(root_path)
        train_data = GraphCount(root=root_path, split="train", task=self.name)
        validation_data = GraphCount(root=root_path, split="val", task=self.name)
        test_data = GraphCount(root=root_path, split="test", task=self.name)
        all_data = torch_geometric.data.InMemoryDataset.collate([train_data._data, validation_data._data, test_data._data])
        self.processed_dataset = all_data[0]
        # flatten y if self.name is not 'multi'
        if self.name != 'multi':
            self.processed_dataset.y = self.processed_dataset.y.view(-1)
        # merge the slices
        self.slices = dict()
        for key in train_data.slices.keys():
            validation_data.slices[key] += train_data.slices[key][-1]
            test_data.slices[key] += validation_data.slices[key][-1]
            self.slices[key] = torch.cat((train_data.slices[key], validation_data.slices[key][1:], test_data.slices[key][1:]))

        self.processed_dataset.primary_node_labels = self.processed_dataset.x
        self.processed_dataset.node_attributes = torch.Tensor()
        self.processed_dataset.primary_edge_labels = torch.Tensor()
        self.processed_dataset.edge_attributes = torch.Tensor()

        self.slices['primary_node_labels'] = self.slices['x']


        sizes = {'num_edge_attributes': 0,
                 'num_edge_labels': 0,
                 'num_node_attributes': 0,
                 'num_node_labels': 0
                 }
        return self.processed_dataset, self.slices, sizes


class GraphCount(InMemoryDataset):

    task_index = dict(
        triangle=0,
        tri_tail=1,
        star=2,
        cycle4=3,
        cycle5=4,
        cycle6=5,
        multi = -1,
    )

    def __init__(self, root:str, split:str, task:str, **kwargs):
        super().__init__(root=root, **kwargs)

        _pt = dict(zip(["train", "val", "test"], self.processed_paths))
        self.data, self.slices = torch.load(_pt[split])

        index = self.task_index[task]
        if index != -1:
            self.data.y = self.data.y[:, index:index+1]

    @property
    def raw_file_names(self):
        return ["Data/GraphDatasets/SubstructureCountingBenchmark.pt"]

    @property
    def processed_dir(self):
        return f"{self.root}/randomgraph"

    @property
    def processed_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]

    def process(self):

        _pt, = self.raw_file_names
        raw = torch.load(f"{self.root}/{_pt}")

        def to(graph):

            A = graph["A"]
            y = graph["y"]

            return Data(
                x=torch.ones(A.shape[0], 1, dtype=torch.int64), y=y,
                edge_index=torch.Tensor(np.vstack(np.where(graph["A"] > 0)))
                     .type(torch.int64),
            )

        data = [to(graph) for graph in raw["data"]]

        if self.pre_filter is not None:
            data = filter(self.pre_filter, data)

        if self.pre_transform is not None:
            data = map(self.pre_transform, data)

        data_list = list(data)
        normalize = torch.std(torch.stack([data.y for data in data_list]), dim=0)

        for split in ["train", "val", "test"]:

            from operator import itemgetter
            split_idx = raw["index"][split]
            splits = itemgetter(*split_idx)(data_list)

            data, slices = self.collate(splits)
            data.y = data.y / normalize

            torch.save((data, slices), f"{self.processed_dir}/{split}.pt")
