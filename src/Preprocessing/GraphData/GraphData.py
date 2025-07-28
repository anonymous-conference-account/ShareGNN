import os
from pathlib import Path
from typing import Dict, Optional, Callable, List, Union

import networkx as nx
import numpy as np
import torch
import torch_geometric.data
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import ZINC, TUDataset, GNNBenchmarkDataset, LRGBDataset

from src.Preprocessing.GraphData.GraphDataPreprocessing import ZINCGraphDataPreprocessing, QMGraphDataPreprocessing, \
    OGBGraphPropertyGraphDataPreprocessing, SubstructureBenchmarkPreprocessing
from src.utils.GraphLabels import NodeLabels, EdgeLabels, Properties
from src.utils.utils import load_graphs
from torch_geometric.io import fs
from torch_geometric.utils.convert import to_networkx
from ogb.nodeproppred import PygNodePropPredDataset


class ShareGNNDataset(InMemoryDataset):
    def __init__(
            self,
            root: str,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            from_existing_data: Union[str, list, None] = None,
            force_reload: bool = False,
            precision: str = 'float',
            input_features = None,
            output_features = None,
            task = None,
            merge_graphs = None,
            experiment_config=None
    ) -> None:
        self.name = name # name of the dataset
        self.from_existing_data = from_existing_data # create the dataset from existing data
        self.nx_graphs = None # networkx graphs
        self.unique_node_labels = 0 # number of unique node labels
        self.node_labels = {} # different node labels for the graph data
        self.edge_labels = {} # different edge labels for the graph data
        self.properties = {} # different pairwise properties for the graph data
        self.precision = torch.float
        self.task = task
        self.experiment_config = experiment_config
        if precision == 'double':
            self.precision = torch.double
        super(ShareGNNDataset, self).__init__(root, transform, pre_transform, force_reload=force_reload)
        out = fs.torch_load(self.processed_paths[0])

        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        assert len(out) == 3 or len(out) == 4


        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        # check if the data object matches our format
        self.validate_dataset_object(data)

        self._num_graph_nodes = torch.zeros(len(self), dtype=torch.long)
        num_node_attributes = self.num_node_attributes
        num_edge_attributes = self.num_edge_attributes

        self.node_labels['primary'] = data['primary_node_labels']
        self.edge_labels['primary'] = data['primary_edge_labels']

        # TODO : do the following in the preprocess function (use primary node labels and primary edge labels and node_attributes and edge_attributes)
        # split node labels and attributes as well as edge labels and attributes
        #num_zero_columns = 0
        #if data.get('x', None) is not None:
        #    if delete_zero_columns and data['x'].layout != torch.sparse_csr:
        #        # if x is one dimensional, add a dimension
        #        if data['x'].dim() == 1:
        #            data['x'] = data['x'].unsqueeze(1)
        #        columns = data['x'].shape[1]
        #        # remove columns with only zeros
        #        if self.precision == torch.float:
        #            data['x'] = data['x'][:, data['x'].sum(dim=0) != 0].float()
        #        else:
        #            data['x'] = data['x'][:, data['x'].sum(dim=0) != 0].double()
        #        num_zero_columns = columns - data['x'].shape[1]
        #        if self.task == 'graph_classification' or self.task == 'graph_regression':
        #            self.sizes['num_node_labels'] = data['x'].shape[1]
        #else:
        #    if data.get('num_nodes', None) is None:
        #        data['num_nodes'] = torch.zeros(len(self), dtype=torch.long)
        #    # create data['x'] using vectors of ones
        #    data['x'] = torch.ones(data['num_nodes'], 1, dtype=self.precision)
        #    self.sizes['num_node_labels'] = 1
        #    self.slices['x'] = [0]
        #    self.slices['x'] += data['_num_nodes']
        #    self.slices['x'] = torch.tensor(self.slices['x'], dtype=torch.long).cumsum(dim=0)

        #if len(data['x'].shape) == 1:
        #    data['x'] = data['x'].unsqueeze(1)
        #if data['x'].shape[1] == 1:
        #    self.node_labels['primary'] = data['x'].clone().detach().long()
        #else:
        #    if self.task == 'graph_classification' or self.task == 'graph_regression':
        #        if data['x'].shape[1] - (num_node_attributes - num_zero_columns) == 1:
        #            self.node_labels['primary'] = data['x'][:, -1].clone().detach().long()
        #        else:
        #            self.node_labels['primary'] = torch.argmax(data['x'][:, num_node_attributes:], dim=1)
        #    if self.task == 'node_classification':
        #        self.node_labels['primary'] = data['y']
        #self.unique_node_labels = torch.unique(self.node_labels['primary']).shape[0]
        #if not use_node_attr:
        #    data['x'] = data['x'][:, self.num_node_attributes-num_zero_columns:]
        #    self.sizes['num_node_attributes'] = 0



        #if data.get('edge_attr', None) is not None:
        #    if len(data['edge_attr'].shape) == 1:
        #        # unsqueeze the edge_attr tensor
        #        data['edge_attr'] = data['edge_attr'].unsqueeze(1)
        #    if data['edge_attr'].shape[1] == 1:
        #        self.edge_labels['primary'] = data['edge_attr'].clone().detach().long()
        #    else:
        #        if data['edge_attr'].shape[1] - num_edge_attributes == 1:
        #            self.edge_labels['primary'] = data['edge_attr'][:, -1].clone().detach().long()
        #        else:
        #            self.edge_labels['primary'] = torch.argmax(data['edge_attr'][:, num_edge_attributes:], dim=1)
        #    if not use_edge_attr:
        #        data['edge_attr'] = data['edge_attr'][:, self.num_edge_attributes:]
        #        self.sizes['num_edge_attributes'] = 0

        #if data.get('y', None) is not None:
        if self.task == 'graph_classification':
            # convert y to long
            data['y'] = data['y'].long()
            # flatten y
            data['y'] = data['y'].view(-1)


        if len(self) == 1:
            data['num_nodes'] = torch.tensor([data['x'].shape[0]], dtype=torch.long)
        else:
            # get the number of nodes for each graph (using the slices x differences)
            data['num_nodes'] = self.slices['x'][1:] - self.slices['x'][:-1]


        self.preprocess_share_gnn_data(data, input_features, output_features, task=task)

        self.number_of_output_classes = 0
        # use the task to determine the number of classes
        if self.task == 'graph_classification':
            self.number_of_output_classes = torch.unique(data['y']).shape[0]
        elif self.task == 'graph_regression':
            # if y is 2D-tensor, take the dimension of the second axis as number of output classes
            if data['y'].dim() == 2:
                self.number_of_output_classes = data['y'].shape[1]
            else:
                self.number_of_output_classes = 1
        elif self.task == 'node_classification':
            self.number_of_output_classes = self.num_node_labels
        elif self.task == 'edge_classification':
            self.number_of_output_classes = self.num_edge_labels
        elif self.task == 'link_prediction':
            self.number_of_output_classes = 2
        else:
            raise ValueError('Task not supported')


        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            # split node labels and attributes as well as edge labels and attributes
            self.data = data_cls.from_dict(data)


        assert isinstance(self._data, Data)

    @property
    def raw_dir(self) -> str:
        name = f'raw'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed'
        return os.path.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    @property
    def num_classes(self) -> int:
        return self.number_of_output_classes


    def process(self):
        sizes = None
        if self.from_existing_data is not None:
            # merge the graphs
            if isinstance(self.from_existing_data, list):
                all_x = None
                all_edge_indices = None
                all_edge_atr = None
                all_y = None
                all_num_nodes = None
                self.slices = {
                    'x': [0],
                    'edge_index': [0],
                    'edge_attr': [0],
                    'y': [0]
                }
                sizes = {
                    'num_node_labels': 0,
                    'num_node_attributes': 0,
                    'num_edge_labels': 0,
                    'num_edge_attributes': 0
                }
                for i, dataset in enumerate(self.from_existing_data):
                    current_x = dataset.data.x
                    current_edge_indices = dataset.data.edge_index
                    current_edge_atr = dataset.data.edge_attr
                    current_y = dataset.data.y
                    current_num_nodes = dataset.data.num_nodes
                    if i == 0:
                        all_x = current_x
                        all_edge_indices = current_edge_indices
                        all_edge_atr = current_edge_atr
                        all_y = current_y
                        all_num_nodes = current_num_nodes
                        self.slices = {
                            'x': dataset.slices['x'],
                            'edge_index': dataset.slices['edge_index'],
                            'y': dataset.slices['y'],
                            'names': [dataset.name] * len(dataset)
                        }
                        if 'edge_attr' in dataset.slices:
                            self.slices['edge_attr'] = dataset.slices['edge_attr']
                        sizes = {
                            'num_node_labels': dataset.num_node_labels,
                            'num_node_attributes': dataset.num_node_attributes,
                            'num_edge_labels': dataset.num_edge_labels,
                            'num_edge_attributes': dataset.num_edge_attributes,
                        }
                    else:
                        max_node_labels = max(sizes['num_node_labels'], dataset.num_node_labels)
                        max_node_attrs = max(sizes['num_node_attributes'], dataset.num_node_attributes)
                        max_edge_labels = max(sizes['num_edge_labels'], dataset.num_edge_labels)
                        max_edge_attrs = max(sizes['num_edge_attributes'], dataset.num_edge_attributes)

                        self.slices['x'] =  torch.cat((self.slices['x'], dataset.slices['x'][1:] + all_x.shape[0]), dim=0)
                        self.slices['edge_index'] = torch.cat((self.slices['edge_index'], dataset.slices['edge_index'][1:] + all_edge_indices.shape[1]), dim=0)
                        if 'edge_attr' in self.slices:
                            self.slices['edge_attr'] = torch.cat((self.slices['edge_attr'], dataset.slices['edge_attr'][1:] + all_edge_atr.shape[0]), dim=0)
                        self.slices['y'] = torch.cat((self.slices['y'], dataset.slices['y'][1:] + all_y.shape[0]), dim=0)

                        # bring all tensors to the same size
                        if sizes['num_node_labels'] < max_node_labels:
                            # add zeros from sizes['num_node_labels'] to max_node_labels
                            all_x = torch.cat((all_x[:, :sizes['num_node_labels']], torch.zeros(all_x.shape[0], max_node_labels - sizes['num_node_labels']), all_x[:, sizes['num_node_labels']:]), dim=1)
                        if dataset.num_node_labels < max_node_labels:
                            current_x = torch.cat((current_x[:, :dataset.num_node_labels], torch.zeros(current_x.shape[0], max_node_labels - dataset.num_node_labels), current_x[:, dataset.num_node_labels:]), dim=1)
                        if sizes['num_node_attributes'] < max_node_attrs:
                            all_x = torch.cat((all_x, torch.zeros(all_x.shape[0], max_node_attrs - sizes['num_node_attributes'])), dim=1)
                        if dataset.num_node_features < max_node_attrs:
                            current_x = torch.cat((current_x, torch.zeros(current_x.shape[0], max_node_attrs - dataset.num_node_features)), dim=1)
                        all_x = torch.cat((all_x, current_x), dim=0)
                        if sizes['num_edge_labels'] < max_edge_labels and all_edge_atr is not None:
                            all_edge_atr = torch.cat((all_edge_atr[:, :sizes['num_edge_labels']], torch.zeros(all_edge_atr.shape[0], max_edge_labels - sizes['num_edge_labels']), all_edge_atr[:, sizes['num_edge_labels']:]), dim=1)
                        if dataset.num_edge_labels < max_edge_labels:
                            current_edge_atr = torch.cat((current_edge_atr[:, :dataset.num_edge_labels], torch.zeros(current_edge_atr.shape[0], max_edge_labels - dataset.num_edge_labels), current_edge_atr[:, dataset.num_edge_labels:]), dim=1)
                        if sizes['num_edge_attributes'] < max_edge_attrs:
                            all_edge_atr = torch.cat((all_edge_atr, torch.zeros(all_edge_atr.shape[0], max_edge_attrs - sizes['num_edge_attributes'])), dim=1)
                        if dataset.num_edge_attributes < max_edge_attrs:
                            current_edge_atr = torch.cat((current_edge_atr, torch.zeros(current_edge_atr.shape[0], max_edge_attrs - dataset.num_edge_attributes)), dim=1)
                        if 'edge_attr' in self.slices:
                            all_edge_atr = torch.cat((all_edge_atr, current_edge_atr), dim=0)
                            all_edge_atr = torch.cat((all_edge_atr, current_edge_atr), dim=0)
                        all_edge_indices = torch.cat((all_edge_indices, current_edge_indices), dim=1)
                        all_y = torch.cat((all_y, current_y), dim=0)
                        all_num_nodes = torch.cat((all_num_nodes, current_num_nodes), dim=0)

                        sizes['num_node_labels'] = max_node_labels
                        sizes['num_node_attributes'] = max_node_attrs
                        sizes['num_edge_labels'] = max_edge_labels
                        sizes['num_edge_attributes'] = max_edge_attrs
                        # make self data from all_x, all_edge_indices, all_edge_atr, all_y
                        self.data = Data(x=all_x, edge_index=all_edge_indices, edge_attr=all_edge_atr, y=all_y, num_nodes=all_num_nodes)




            elif self.from_existing_data in ['ZINC', 'ZINC-full', 'ZINC-Full', 'ZINCFull', 'ZINC-12k', 'ZINC-25k']:
                preprocessed_data = ZINCGraphDataPreprocessing(self.name)
                self.data, self.slices, sizes = preprocessed_data.processed_dataset, preprocessed_data.slices, preprocessed_data.sizes
                pass
            elif self.from_existing_data in ['QM9', 'QM-9', 'QM7', 'QM-7', 'QM8', 'QM-8']:
                preprocessed_data = QMGraphDataPreprocessing(self.name)
                self.data, self.slices, sizes = preprocessed_data.processed_dataset, preprocessed_data.slices, preprocessed_data.sizes
                pass
            elif self.from_existing_data == 'OGB_GraphProp':
                preprocessed_data = OGBGraphPropertyGraphDataPreprocessing(self.name)
                self.data, self.slices, sizes = preprocessed_data.processed_dataset, preprocessed_data.slices, preprocessed_data.sizes
                pass
            elif self.from_existing_data == 'MoleculeNet':
                dataset = torch_geometric.datasets.MoleculeNet(root='tmp/', name=self.name)
                self.data = dataset.data
                # put column 0 of x and edge_attr to the end
                self.data.x = torch.cat((self.data.x[:, 1:], self.data.x[:, 0].unsqueeze(1)), dim=1)
                unique_node_labels = torch.unique(self.data.x[:, -1])
                # map unique node labels to integers 0, 1, 2, ...
                self.data.x[:, -1] = torch.tensor([torch.where(unique_node_labels == x)[0] for x in self.data.x[:, -1]], dtype=torch.long)
                self.data.edge_attr = torch.cat((self.data.edge_attr[:, 1:], self.data.edge_attr[:, 0].unsqueeze(1)), dim=1)
                unique_edge_labels = torch.unique(self.data.edge_attr[:, -1])
                # map unique edge labels to integers 0, 1, 2, ...
                self.data.edge_attr[:, -1] = torch.tensor([torch.where(unique_edge_labels == x)[0] for x in self.data.edge_attr[:, -1]], dtype=torch.long)
                self.slices = dataset.slices
                sizes = {
                    'num_node_labels': torch.unique(dataset.data.x[:,0]),
                    'num_node_attributes': 8,
                    'num_edge_labels': len(torch.unique(dataset.data.edge_attr[:, 0])),
                    'num_edge_attributes': 2,
                }
                pass
            elif self.from_existing_data == 'SubstructureBenchmark':
                preprocessed_data = SubstructureBenchmarkPreprocessing(self.name)
                self.data, self.slices, sizes = preprocessed_data.processed_dataset, preprocessed_data.slices, preprocessed_data.sizes


                pass
            elif self.from_existing_data in ['planetoid', 'cora', 'citeseer', 'pubmed', 'Planetoid']:
                dataset = torch_geometric.datasets.Planetoid(root='tmp/', name=self.name)
                self.data = dataset[0]
                self.slices = dict()
                for key, value in dataset.data:
                    self.slices[key] = torch.tensor([0, value.shape[0]], dtype=torch.long)
                sizes = {
                    'num_node_labels': len(torch.unique(self.data.y)),
                    'num_node_attributes': self.data.x.shape[1],
                    'num_edge_labels': 0,
                    'num_edge_attributes': 0
                }
            elif self.from_existing_data in ['Nell', 'nell', 'NELL']:
                dataset = torch_geometric.datasets.NELL(root='tmp/')
                self.data = dataset[0]
                self.slices = dict()
                for key, value in dataset.data:
                    self.slices[key] = torch.tensor([0, value.shape[0]], dtype=torch.long)
                sizes = {
                    'num_node_labels': len(torch.unique(self.data.y)),
                    'num_node_attributes': self.data.x.shape[1],
                    'num_edge_labels': 0,
                    'num_edge_attributes': 0
                }
            elif self.from_existing_data in ['ogbn', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M', 'ogbn-mag']:
                dataset = PygNodePropPredDataset(name=self.name, root='tmp/')
                split_idx = dataset.get_idx_split()
                train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
                self.data = dataset[0]  # pyg graph object
                self.data.train_mask = self.data.val_mask = self.data.test_mask = None
                self.data.train_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
                self.data.train_mask[train_idx] = 1
                self.data.val_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
                self.data.val_mask[valid_idx] = 1
                self.data.test_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
                self.data.test_mask[test_idx] = 1
                self.slices = dict()
                for key, value in self.data:
                    if isinstance(value, torch.Tensor):
                        self.slices[key] = torch.tensor([0, value.shape[0]], dtype=torch.long)
                sizes = {
                    'num_node_labels': len(torch.unique(self.data.y)),
                    'num_node_attributes': self.data.x.shape[1],
                    'num_edge_labels': 0,
                    'num_edge_attributes': 0
                }

            elif self.from_existing_data == 'TUDataset':
                tu_dataset = TUDataset(root='tmp/', name=self.name, use_node_attr=True, use_edge_attr=True)
                if 'x' not in tu_dataset.data:
                    tu_dataset.data.x = torch.zeros((tu_dataset.num_nodes,1), dtype=torch.float)
                    tu_dataset.data.primary_node_labels = torch.zeros(tu_dataset.num_nodes, dtype=torch.long)
                    tu_dataset.slices['x'] = torch.zeros(len(tu_dataset)+1, dtype=torch.long)
                    # get slices from edge_index_slices
                    edge_slices = tu_dataset.slices['edge_index']
                    for i, edge_slice in enumerate(edge_slices):
                        if i > 0:
                            start = edge_slices[i-1]
                            end = edge_slices[i]
                            num_nodes = torch.max(tu_dataset.edge_index[:, start:end]) - torch.min(tu_dataset.edge_index[:, start:end]) + 1
                            tu_dataset.slices['x'][i] = tu_dataset.slices['x'][i-1] + num_nodes
                else:
                    tu_dataset.data.primary_node_labels = torch.argmax(tu_dataset.data.x[:,tu_dataset.sizes['num_node_attributes']:], dim=1)
                tu_dataset.slices['primary_node_labels'] = tu_dataset.slices['x']
                if tu_dataset.sizes['num_node_attributes'] > 0:
                    tu_dataset.data.node_attributes = tu_dataset.data.x[:,:tu_dataset.sizes['num_node_attributes']]
                    tu_dataset.slices['node_attributes'] = tu_dataset.slices['x']
                else:
                    tu_dataset.data.node_attributes = torch.Tensor()
                if tu_dataset.data.edge_attr is None:
                    tu_dataset.data.primary_edge_labels = torch.Tensor()
                    tu_dataset.data.edge_attributes = torch.Tensor()
                else:
                    tu_dataset.data.primary_edge_labels = torch.argmax(tu_dataset.data.edge_attr[:,tu_dataset.sizes['num_edge_attributes']:], dim=1)
                    tu_dataset.slices['primary_edge_labels'] = tu_dataset.slices['edge_attr']
                    tu_dataset.data.edge_attributes = tu_dataset.data.edge_attr[:,:tu_dataset.sizes['num_edge_attributes']]
                    tu_dataset.slices['edge_attributes'] = tu_dataset.slices['edge_attr']
                self.data, self.slices, sizes = tu_dataset._data, tu_dataset.slices, tu_dataset.sizes
            elif self.from_existing_data == 'NEL':
                self.data, self.slices, sizes = self.read_nel_data_v2()
            elif self.from_existing_data == 'gnn_benchmark':
                dataset = GNNBenchmarkDataset("tmp/", self.name)
                sizes = {
                    'num_node_labels': dataset.num_features,
                    'num_node_attributes': dataset.num_node_features,
                    'num_edge_labels': dataset.num_edge_features,
                    'num_edge_attributes': 0
                }
                # add x to data uing num_nodes times 0
                num_graphs = len(dataset.data.y)
                dataset.data.x = torch.ones(dataset.data.num_nodes, 1)
                nodes_per_graph = dataset.data.num_nodes // num_graphs
                # remove num_nodes from x
                dataset.slices['x'] = torch.linspace(0, dataset.data.num_nodes, num_graphs + 1, dtype=torch.long)
                dataset.data.primary_node_labels = torch.zeros(dataset.data.num_nodes, dtype=torch.long)
                dataset.data.node_attributes = torch.Tensor()
                dataset.data.primary_edge_labels = torch.Tensor()
                dataset.data.edge_attributes = torch.Tensor()
                dataset.slices['primary_node_labels'] = dataset.slices['x']
                self.slices = dataset.slices
                self.data = dataset.data
                pass
            elif self.from_existing_data == 'Peptides':
                dataset = torch_geometric.datasets.LRGBDataset(root='tmp/', name=self.name)
                self.data = dataset.data
                self.slices = dataset.slices
                sizes = {
                    'num_node_labels': dataset.num_node_features,
                    'num_node_attributes': dataset.num_node_features,
                    'num_edge_labels': dataset.num_edge_features,
                    'num_edge_attributes': dataset.num_edge_features
                }
        else:
            print('Cannot process the data')

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self._data, Data)
        fs.torch_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )

    def validate_dataset_object(self, data):
        # check whether there is at least node_labels, node_attributes, edge_labels, edge_attributes, x, y
        if not 'x' in data:
            raise ValueError("Data object must have an attribute 'x' for node features.")
        if not 'x' in self.slices:
            raise ValueError("Data object must have an attribute 'x' in slices for node features.")
        if not 'y' in data:
            raise ValueError("Data object must have an attribute 'y' for labels.")
        if not 'y' in self.slices:
            raise ValueError("Data object must have an attribute 'y' in slices for labels.")
        if not 'edge_index' in data:
            raise ValueError("Data object must have an attribute 'edge_index' for edge indices.")
        if not 'edge_index' in self.slices:
            raise ValueError("Data object must have an attribute 'edge_index' in slices for edge indices.")
        if not 'primary_node_labels' in data:
            raise ValueError("Data object must have an attribute 'primary_node_labels' for node labels.")
        if not 'primary_node_labels' in self.slices:
            raise ValueError("Data object must have an attribute 'primary_node_labels' in slices for node labels.")
        if not 'primary_edge_labels' in data:
            raise ValueError("Data object must have an attribute 'primary_edge_labels' for edge labels.")
        if not 'node_attributes' in data:
            raise ValueError("Data object must have an attribute 'node_attributes' for node attributes.")
        if not 'edge_attributes' in data:
            raise ValueError("Data object must have an attribute 'edge_attributes' for edge attributes.")
        pass

    def read_nel_data(self):
        graphs, labels = load_graphs(Path(self.raw_dir), self.name, graph_format='NEL')
        node_labels = []
        node_attributes = []
        node_slices = [0]
        with_node_attributes = False
        for graph_id, graph in enumerate(graphs):
            print(f'Processing graph {graph_id+1}/{len(graphs)}')
            node_labels += [0] * graph.number_of_nodes()
            node_attributes += [0] * graph.number_of_nodes()
            for node in graph.nodes(data=True):
                if 'primary_node_labels' in node[1]:
                    index_start = np.sum(node_slices[0:graph_id+1])
                    node_labels[index_start+node[0]] = int(node[1]['primary_node_labels'][0])
                    if len(node[1]['primary_node_labels']) > 1:
                        with_node_attributes = True
                        node_attributes[index_start+node[0]] = node[1]['primary_node_labels'][1:]
            node_slices.append(graph.number_of_nodes())
        # convert the node labels to a tensor
        node_labels = torch.tensor(node_labels, dtype=torch.long)
        # apply row-wise one-hot encoding
        node_labels = torch.nn.functional.one_hot(node_labels).float()
        # convert the node attributes to a tensor
        node_attributes = torch.tensor(node_attributes, dtype=torch.float)
        if len(node_attributes) == 0 or not with_node_attributes:
            node_attributes = None
        if node_attributes is not None:
            # stack node attributes and node labels together to form the node feature matrix
            x = torch.cat((node_attributes, node_labels), dim=1)
        else:
            x = node_labels
        node_slices = torch.tensor(node_slices, dtype=torch.long).cumsum(dim=0)
        # create edge_index tensor
        edge_indices = []
        edge_slices = [0]
        edge_labels = []
        edge_attributes = []
        for i, graph in enumerate(graphs):
            for edge in graph.edges(data=True):
                edge_indices.append([edge[0], edge[1]])
                if 'primary_edge_labels' in edge[2]:
                    edge_labels.append(int(edge[2]['primary_edge_labels'][0]))
                    if len(edge[2]['primary_edge_labels']) > 1:
                        edge_attributes.append(edge[2]['primary_edge_labels'][1:])
            edge_slices.append(len(graph.edges()))
        # convert the edge indices to a tensor
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).T
        edge_slices = torch.tensor(edge_slices, dtype=torch.long).cumsum(dim=0)
        # convert the edge labels to a tensor
        edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        # apply row-wise one-hot encoding
        edge_labels = torch.nn.functional.one_hot(edge_labels).float()
        # convert the edge attributes to a tensor
        edge_attributes = torch.tensor(edge_attributes, dtype=torch.float)
        if len(edge_attributes) == 0:
            edge_attributes = None
        if edge_attributes is not None:
            # stack edge attributes and edge labels together to form the edge feature matrix
            edge_attr = torch.cat((edge_attributes, edge_labels), dim=1)
        else:
            edge_attr = edge_labels
        y = torch.tensor(labels, dtype=torch.long)
        y_slices = torch.arange(0, len(labels)+1, dtype=torch.long)
        data = Data(x=x, edge_index=edge_indices, edge_attr=edge_attr, y=y)
        slices = {'edge_index': edge_slices,
                    'x': node_slices,
                  'edge_attr': edge_slices.detach().clone(),
                  'y': y_slices}
        sizes = {'num_node_labels': node_labels.shape[1],
                 'num_node_attributes': node_attributes.shape[1] if node_attributes is not None else 0,
                 'num_edge_labels': edge_labels.shape[1],
                 'num_edge_attributes': edge_attributes.shape[1] if edge_attributes is not None else 0}
        return data, slices, sizes

    def read_nel_data_v2(self):
        load_path = Path(self.raw_dir)
        # load the nodes from the file
        node_labels = []
        node_attributes = None
        node_slices = [0]
        node_counter = 0
        unique_labels = []
        num_graphs = 0
        with open(load_path.joinpath(self.name + "_Nodes.txt"), "r") as f:
            lines = f.readlines()
            line_length = len(lines[0].strip().split(" "))
            # convert into torch tensor
            torch_lines = torch.zeros((len(lines), line_length), dtype=torch.float)
            for i, line in enumerate(lines):
                if i % 10000 == 0:
                    print(f'Processing node {i+1}/{len(lines)} in dataset {self.name}')
                line_data = line.strip().split(" ")
                torch_lines[i] = torch.tensor(list(map(float, line_data)))
            graph_ids = torch_lines[:, 0].long()
            # get slice vector from unique graph ids
            node_slices = torch.unique(graph_ids, return_counts=True)[1]
            num_graphs = len(node_slices)
            # add 0 at the beginning
            node_slices = torch.cat((torch.tensor([0]), node_slices)).cumsum(dim=0).long()
            node_ids = torch_lines[:, 1].long()
            node_labels = torch_lines[:, 2].long()
            unique_labels = len(torch.unique(node_labels))
            node_attr = None
            # sort the node labels graph-wise (node_slices) according to the node ids
            for idx in range(len(node_slices) - 1):
                sorted_indices = torch.argsort(node_ids[node_slices[idx]:node_slices[idx+1]])
                node_labels[node_slices[idx]:node_slices[idx+1]] = node_labels[node_slices[idx]:node_slices[idx+1]][sorted_indices]
                if line_length > 3:
                    node_attr = torch_lines[:, 3:]
                    node_attr[node_slices[idx]:node_slices[idx+1]] = node_attr[node_slices[idx]:node_slices[idx+1]][sorted_indices]

        x = None
        # one hot encoding if number of node labels is smaller than 100
        if unique_labels < 100:
            x = torch.nn.functional.one_hot(node_labels).float()
        else:
            x = node_labels
        if node_attr is not None:
            x = torch.cat((node_attr, x), dim=1)

        edge_indices = None
        edge_slices = None
        edge_labels = None
        edge_attr = None
        with open(load_path.joinpath(self.name + "_Edges.txt"), "r") as f:
            lines = f.readlines()
            line_length = len(lines[0].strip().split(" "))
            torch_lines = torch.zeros((len(lines), line_length), dtype=torch.float)
            for i, line in enumerate(lines):
                if i % 10000 == 0:
                    print(f'Processing edge {i+1}/{len(lines)} in dataset {self.name}')
                data = line.strip().split(" ")
                torch_lines[i] = torch.tensor(list(map(float, data)))
            graph_ids = torch_lines[:, 0].long()
            all_ids = torch.unique(graph_ids)
            # missing ids are those not in all_ids but in range(0, num_graphs)
            missing_ids = [i for i in range(num_graphs) if i not in all_ids]
            # get slice vector from unique graph ids
            edge_slices = torch.unique(graph_ids, return_counts=True)[1]
            # add 0 at the beginning
            edge_slices = torch.cat((torch.tensor([0]), edge_slices)).cumsum(dim=0).long()
            edge_slices = [x.item() for x in edge_slices]
            # duplicate the value at the index of the missing ids
            for missing_id in missing_ids:
                edge_slices.insert(missing_id+1, edge_slices[missing_id])
            edge_slices = torch.tensor(edge_slices, dtype=torch.long)
            edge_indices = torch_lines[:, 1:3].long().T
            edge_labels = torch_lines[:, 3].long()
            edge_labels = torch.nn.functional.one_hot(edge_labels).float()
            edge_attr = None
            if line_length > 4:
                edge_attr = torch_lines[:, 4:]
                edge_data = torch.cat((edge_attr, edge_labels), dim=1)
            else:
                edge_data = edge_labels

        y = None
        with open(load_path.joinpath(self.name + "_Labels.txt"), "r") as f:
            lines = f.readlines()
            line_length = len(lines[0].strip().split(" "))
            torch_lines = torch.zeros((len(lines), line_length - 1), dtype=torch.long)
            for i, line in enumerate(lines):
                data = line.strip().split(" ")
                graph_name = data[0]
                torch_lines[i] = torch.tensor(list(map(float, data[1:])))
            y = torch_lines[:, 1].long()


        y_slices = torch.arange(0, len(y) + 1, dtype=torch.long)
        data = Data(x=x, edge_index=edge_indices, edge_attr=edge_data, y=y)
        slices = {'edge_index': edge_slices,
                  'x': node_slices,
                  'edge_attr': edge_slices.detach().clone(),
                  'y': y_slices}

        sizes = {'num_node_labels': unique_labels,
                 'num_node_attributes': node_attributes.shape[1] if node_attributes is not None else 0,
                 'num_edge_labels': edge_labels.shape[1],
                 'num_edge_attributes': edge_attr.shape[1] if edge_attr is not None else 0}
        data.primary_node_labels = node_labels
        slices['primary_node_labels'] = slices['x']
        if node_attributes is not None:
            data.node_attributes = node_attributes
            slices['node_attributes'] = slices['x']
        else:
            data.node_attributes = torch.Tensor()
        if len(torch.unique(edge_labels)) > 1:
            data.primary_edge_labels = data.edge_labels.long()
            slices['primary_edge_labels'] = slices['edge_attr']
        else:
            data.primary_edge_labels = torch.Tensor()
        if edge_attr is not None:
            data.edge_attributes = edge_attr
            slices['edge_attributes'] = slices['edge_attr']
        else:
            data.edge_attributes = torch.Tensor()

        return data, slices, sizes

    def create_nx_graph(self, graph_id: int, directed: bool = False):
        graph = self[graph_id]
        nx_graph = to_networkx(
                data=graph,
                node_attrs=['primary_node_labels'] if self.task != 'node_classification' else None,
                edge_attrs=['primary_edge_labels'] if 'primary_edge_labels' in graph else None,
                to_undirected=not directed)
        return nx_graph

    def create_nx_graphs(self, directed: bool = False):
        self.nx_graphs = []
        counter = 0
        for g_id, graph in enumerate(self):
            if g_id % 1000 == 0:
                print(f'Processing graph {g_id+1}/{len(self)}')

            self.nx_graphs.append(to_networkx(
                data=graph,
                node_attrs=['primary_node_labels'] if self.task != 'node_classification' else None,
                edge_attrs=['primary_edge_labels'] if 'primary_edge_labels' in graph else None,
                to_undirected=not directed))
        pass

    def preprocess_share_gnn_data(self, data, input_features=None, output_features=None, task=None) -> None:
        if input_features is not None and task is not None:
            use_labels = input_features.get('name', 'node_labels') == 'node_labels'
            use_constant = input_features.get('name', 'node_labels') == 'constant'
            use_features = input_features.get('name', 'node_labels') == 'node_features'
            use_labels_and_features = input_features.get('name', 'node_labels') == 'all'
            transformation = input_features.get('transformation', None)
            use_features_as_channels = input_features.get('features_as_channels', False)

            use_train_node_labels = (task == 'node_classification') and input_features.get('one_hot_train_labels', False)

            ### Determine the input data
            if use_labels:
                data['x'] = data['primary_node_labels'].long()
                if transformation in ['one_hot', 'one_hot_encoding']:
                    # one hot encode the node labels
                    data['x'] = torch.nn.functional.one_hot(data['x'])
                    if data['x'].shape[1] == 1:
                        data['x'] = data['x'].squeeze(1)
                data['x'] = data['x'].type(self.precision)
            elif use_constant:
                data['x'] = torch.full(size=(data['x'].shape[0], input_features.get('in_dimensions', 1)), fill_value=input_features.get('value', 1.0), dtype=self.precision)
            elif use_features:
                data['x'] = data['node_attributes'].type(self.precision)
                if use_train_node_labels:
                    # get data y one hot
                    y_one_hot = torch.nn.functional.one_hot(data['y']).type(self.precision)
                    # set all rows of y_one_hot to 1/row_num if row is not in train mask
                    non_train_indices = (data['train_mask'] == 0).nonzero().squeeze()
                    y_one_hot[non_train_indices] = 1.0 / y_one_hot.shape[1]
                    data['x'] = torch.cat((y_one_hot, data['x']), dim=1)
            elif use_labels_and_features:
                # get first self.num_node_attributes columns and on the rest apply argmax
                data['x'] = torch.cat((data['node_attributes'].type(self.precision), data['primary_node_labels'].long().unsqueeze(1)), dim=1)
            else:
                pass

            # normalize the graph input labels, i.e. to have values between -1 and 1, no zero values
            if use_labels and transformation == 'normalize':
                # get the number of unique node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # if value is even, add 1 to make it odd
                    if value % 2 == 0:
                        value = ((value + 1) * interval_length)
                    else:
                        value = (-1) * (value * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values
                data['x'] = data['x'].apply_(lambda x: normalized_node_labels[x])
            elif use_labels and transformation == 'normalize_positive':
                # get the number of different node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # map the value to the interval [0,1]
                    value = ((value + 1) * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values
                data['x'] = data['x'].apply_(lambda x: normalized_node_labels[x])
            elif use_labels and transformation == 'unit_circle':
                '''
                Arrange the labels in an 2D unit circle
                '''
                num_node_labels = self.unique_node_labels
                # duplicate data column
                data['x'] = data['x'].repeat(1, 2)
                data['x'] = data['x'][:, 0:1].apply_(lambda x: torch.cos(2 * np.pi * x / num_node_labels))
                data['x'] = data['x'][:, 1:2].apply_(lambda x: torch.sin(2 * np.pi * x / num_node_labels))
            elif use_labels_and_features and transformation == 'one_hot_labels_normalize_features':
                one_hot_labels = data['primary_node_labels'].long()
                max_label = torch.max(one_hot_labels)
                one_hot_labels = torch.nn.functional.one_hot(one_hot_labels, num_classes=max_label + 1).type(self.precision)
                # find non-zero columns
                non_zero_columns = torch.where(one_hot_labels.sum(dim=0) != 0)[0]
                one_hot_labels = one_hot_labels[:, non_zero_columns]
                normalized_features = data['node_attributes'].type(self.precision)
                non_zero_columns = torch.where(normalized_features.sum(dim=0) != 0)[0]
                normalized_features = normalized_features[:, non_zero_columns]
                normalized_features = (normalized_features - normalized_features.min(dim=0, keepdim=True).values) / (normalized_features.max(dim=0, keepdim=True).values - normalized_features.min(dim=0, keepdim=True).values)
                data['x'] = torch.cat((one_hot_labels, normalized_features), dim=1)
            elif use_labels_and_features and transformation == 'normalize_labels':
                # get the number of unique node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # if value is even, add 1 to make it odd
                    if value % 2 == 0:
                        value = ((value + 1) * interval_length)
                    else:
                        value = (-1) * (value * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values only for the last column
                data['x'] = data['x'][:, -1].apply_(lambda x: normalized_node_labels[x])
            elif use_labels_and_features and transformation == 'normalize_positive':
                # get the number of different node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # map the value to the interval [0,1]
                    value = ((value + 1) * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values only for the last column
                data['x'] = data['x'][:, -1].apply_(lambda x: normalized_node_labels[x])


            # Determine the output data
            #if task == 'regression':
            #    self.num_classes = 1
            #    if type(self.graph_labels[0]) == list:
            #        self.num_classes = len(self.graph_labels[0])
            #else:
            #    try:
            #        self.num_classes = len(set(self.graph_labels))
            #    except:
            #        self.num_classes = len(self.graph_labels[0])
            #
            # one hot encode y



            if task == 'graph_regression':
                if isinstance(output_features, dict):
                    if output_features.get('transformation', None) is not None:
                        data['y'] = transform_data(data['y'], output_features)

                # select regression task
                if 'regression_targets' in self.experiment_config:
                    if isinstance(self.experiment_config['regression_targets'], list):
                        data['y'] = data['y'][:, self.experiment_config['regression_targets']]
                    elif isinstance(self.experiment_config['regression_targets'], int):
                        data['y'] = data['y'][:, self.experiment_config['regression_targets']:self.experiment_config['regression_targets'] + 1]
                    else:
                        raise ValueError("regression_tasks must be a list of indices")


            elif task == 'node_classification':
                pass
                #data['y'] = torch.nn.functional.one_hot(data['y'], num_classes=self.num_classes).float()
            # if output_normalization is set, normalize the output data dimension-wise
            if isinstance(output_features, dict):
                if output_features.get('normalization', None) is not None:
                    data['original_y'] = data['y'].clone()
                    if output_features.get('normalization', 'standard') == 'standard':
                        for i in range(data['y'].shape[1]):
                            data['y'][:, i] = (data['y'][:, i] - data['y'][:, i].mean()) / (data['y'][:, i].std() + 1e-8)
                    elif output_features.get('normalization', 'standard') == 'minmax':
                        for i in range(data['y'].shape[1]):
                            data['y'][:, i] = (data['y'][:, i] - data['y'][:, i].min()) / (data['y'][:, i].max() - data['y'][:, i].min() + 1e-8)
                    elif output_features.get('normalization', 'standard') == 'minmax_zero':
                        for i in range(data['y'].shape[1]):
                            data['y'][:, i] = (data['y'][:, i] - data['y'][:, i].min()) / (data['y'][:, i].max() - data['y'][:, i].min() + 1e-8)

            return None

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def relabel_most_frequent(labels: NodeLabels, num_max_labels: int):
    if num_max_labels is None:
        num_max_labels = -1
    # get the k most frequent node labels or relabel all
    if num_max_labels == -1:
        bound = len(labels.db_unique_node_labels)
    else:
        bound = min(num_max_labels, len(labels.db_unique_node_labels))
    most_frequent = sorted(labels.db_unique_node_labels, key=labels.db_unique_node_labels.get, reverse=True)[
                    :bound - 1]
    # relabel the node labels
    for i, _lab in enumerate(labels.node_labels):
        for j, lab in enumerate(_lab):
            if lab not in most_frequent:
                labels.node_labels[i][j] = bound - 1
            else:
                labels.node_labels[i][j] = most_frequent.index(lab)
    # set the new unique labels
    labels.num_unique_node_labels = bound
    db_unique = {}
    for i, l in enumerate(labels.node_labels):
        unique = {}
        for label in l:
            if label not in unique:
                unique[label] = 1
            else:
                unique[label] += 1
            if label not in db_unique:
                db_unique[label] = 1
            else:
                db_unique[label] += 1
        labels.unique_node_labels[i] = unique
    labels.db_unique_node_labels = db_unique
    pass


def transform_data(data, transformation_dict: Dict[str, Dict[str, str]]):
    # reformat the output data, shift to positive values and make the values smaller
    transformation_args = [{}]
    if transformation_dict.get('transformation_args', None) is not None:
        transformation_args = transformation_dict['transformation_args']
    if type(transformation_dict['transformation']) == list:
        for i, expression in enumerate(transformation_dict['transformation']):
            data = eval(expression)(input=data, **transformation_args[i])
    else:
        data = eval(transformation_dict['transformation'])(input=data, **transformation_args)
    return data


class GraphData:
    def __init__(self):
        self.name = ''
        self.graphs = []
        self.input_data = []
        self.node_labels: Dict[str, NodeLabels] = {}
        self.edge_labels: Dict[str, EdgeLabels] = {}
        self.properties: Dict[str, Properties] = {}
        self.graph_labels = []
        self.output_data = []
        self.num_classes = 0
        self.max_nodes = 0
        self.num_graphs = 0
        self.input_feature_dimensions = 1
        self.input_channels = 1
        self.output_feature_dimensions = 1
        self.avg_nodes = 0
        self.avg_degree = 0

    def __len__(self):
        return len(self.graphs)

    def __iadd__(self, other):
        '''
        Add another GraphData object to this one.
        '''
        if 'Union' in self.name:
            pass
        else:
            self.name = f'Union_{self.name}'
        self.name += f'_{other.name}'
        self.graphs += other.graphs
        self.input_data += other.input_data

        for key, value in other.node_labels.items():
            if key in self.node_labels:
                self.node_labels[key] += value
            else:
                self.node_labels[key] = value

        for key, value in other.edge_labels.items():
            if key in self.edge_labels:
                self.edge_labels[key] += value
            else:
                self.edge_labels[key] = value

        for key, value in other.properties.items():
            if key in self.properties:
                self.properties[key] += value
            else:
                self.properties[key] = value


        self.graph_labels += other.graph_labels
        self.output_data += other.output_data
        self.num_classes = max(self.num_classes, other.num_classes)
        self.max_nodes = max(self.max_nodes, other.max_nodes)

    def add_node_labels(self, node_labeling_name, max_labels=-1, node_labeling_method=None, **kwargs) -> None:
        if node_labeling_method is not None:
            node_labeling = NodeLabels()
            node_labeling.node_labels, node_labeling.unique_node_labels, node_labeling.db_unique_node_labels = node_labeling_method(
                self.graphs, **kwargs)
            node_labeling.num_unique_node_labels = max(1, len(node_labeling.db_unique_node_labels))

            key = node_labeling_name
            if max_labels is not None and max_labels > 0:
                key = f'{node_labeling_name}_{max_labels}'

            self.node_labels[key] = node_labeling
            relabel_most_frequent(self.node_labels[key], max_labels)

    def add_edge_labels(self, edge_labeling_name, edge_labeling_method=None, **kwargs) -> None:
        if edge_labeling_method is not None:
            edge_labeling = EdgeLabels()
            edge_labeling.edge_labels, edge_labeling.unique_edge_labels, edge_labeling.db_unique_edge_labels = edge_labeling_method(
                self.graphs, **kwargs)
            edge_labeling.num_unique_edge_labels = max(1, len(edge_labeling.db_unique_edge_labels))
            self.edge_labels[edge_labeling_name] = edge_labeling

    def load_nel_graphs(self, db_name: str, path: Path, input_features=None, output_features=None, task=None, only_graphs=False):
        self.name = db_name
        self.graphs, self.graph_labels = load_graphs(path.joinpath(Path(f'{db_name}/raw/')), db_name, graph_format='NEL')
        self.num_graphs = len(self.graphs)
        self.avg_nodes = sum([g.number_of_nodes() for g in self.graphs]) / self.num_graphs
        self.avg_degree = sum([g.number_of_edges() for g in self.graphs]) / self.num_graphs

        self.max_nodes = max([g.number_of_nodes() for g in self.graphs])

        #self.add_node_labels(node_labeling_name='primary', node_labeling_method=NodeLabeling.standard_node_labeling)
        #self.add_edge_labels(edge_labeling_name='primary', edge_labeling_method=EdgeLabeling.standard_edge_labeling)

        if not only_graphs:
            if input_features is None:
                input_features = {'name': 'node_labels', 'transformation': {'name': 'normalize'}}
            if output_features is None:
                output_features = {}

            use_labels = input_features.get('name', 'node_labels') == 'node_labels'
            use_constant = input_features.get('name', 'node_labels') == 'constant'
            use_features = input_features.get('name', 'node_labels') == 'node_features'
            use_labels_and_features = input_features.get('name', 'node_labels') == 'all'
            transformation = input_features.get('transformation', None)

            ### Determine the input data
            self.input_data = []
            ## add node labels
            for graph_id, graph in enumerate(self.graphs):
                if use_labels:
                    if transformation in ['one_hot', 'one_hot_encoding']:
                        self.input_data.append(torch.zeros(1,graph.number_of_nodes(), self.node_labels['primary'].num_unique_node_labels))
                        for node in graph.nodes(data=True):
                            self.input_data[-1][0][node[0]][self.node_labels['primary'].node_labels[graph_id][node[0]]] = 1
                    else:
                        self.input_data.append(torch.ones(1,graph.number_of_nodes(),1).float())
                        for node in graph.nodes(data=True):
                            self.input_data[-1][0][node[0]] = self.node_labels['primary'].node_labels[graph_id][node[0]]
                elif use_constant:
                    self.input_data.append(torch.full(size=(1,graph.number_of_nodes(),1), fill_value=input_features.get('value', 1.0)).float())
                elif use_features:
                    self.input_data.append(torch.zeros(1,graph.number_of_nodes(), len(graph.nodes(data=True)[0]['primary_node_labels'][1:])))
                    for node in graph.nodes(data=True):
                        # add all except the first element of the label
                        self.input_data[-1][0][node[0]] = torch.tensor(node[1]['primary_node_labels'][1:])
                elif use_labels_and_features:
                    self.input_data.append(torch.zeros(1,graph.number_of_nodes(), len(graph.nodes(data=True)[0]['primary_node_labels'])))
                    for node in graph.nodes(data=True):
                        # add all except the first element of the label
                        self.input_data[-1][0][node[0]] = torch.tensor([self.node_labels['primary'].node_labels[graph_id][node[0]]] + node[1]['primary_node_labels'][1:])



            # normalize the graph input labels, i.e. to have values between -1 and 1, no zero values
            if use_labels and transformation == 'normalize':
                # get the number of different node labels
                num_node_labels = self.node_labels['primary'].num_unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                for i, graph in enumerate(self.graphs):
                    for j in range(graph.number_of_nodes()):
                        value = self.input_data[i][0][j]
                        # get integer value of the node label
                        value = int(value)
                        # if value is even, add 1 to make it odd
                        if value % 2 == 0:
                            value = ((value + 1) * interval_length)
                        else:
                            value = (-1) * (value * interval_length)
                        self.input_data[i][0][j] = value
            elif use_labels and transformation == 'normalize_positive':
                # get the number of different node labels
                num_node_labels = self.node_labels['primary'].num_unique_node_labels
                # get the next even number if the number of node labels is odd
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                for i, graph in enumerate(self.graphs):
                    for j in range(graph.number_of_nodes()):
                        value = self.input_data[i][0][j]
                        # get integer value of the node label
                        value = int(value)
                        # map the value to the interval [0,1]
                        value = ((value + 1) * interval_length)
                        self.input_data[i][0][j] = value


            elif use_labels and transformation == 'unit_circle':
                '''
                Arange the labels in an 2D unit circle
                # TODO: implement this
                '''
                updated_input_data = []
                # get the number of different node labels
                num_node_labels = self.node_labels['primary'].num_unique_node_labels
                for i, graph in enumerate(self.graphs):
                    updated_input_data.append(torch.ones(1, graph.number_of_nodes(), 2))
                    for j in range(graph.number_of_nodes()):
                        value = int(self.input_data[i][0][j])
                        # get integer value of the node label
                        value = int(value)
                        updated_input_data[-1][0][j][0] = np.cos(2*np.pi*value / num_node_labels)
                        updated_input_data[-1][0][j][1] = np.sin(2*np.pi*value / num_node_labels)
                self.input_data = updated_input_data
            elif use_labels_and_features and transformation == 'normalize_labels':
                # get the number of different node labels
                num_node_labels = self.node_labels['primary'].num_unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                for i, graph in enumerate(self.graphs):
                    for j in range(graph.number_of_nodes()):
                        value = self.input_data[i][j][0]
                        # get integer value of the node label
                        value = int(value)
                        # if value is even, add 1 to make it odd
                        if value % 2 == 0:
                            value = ((value + 1) * interval_length)
                        else:
                            value = (-1) * (value * interval_length)
                        self.input_data[i][j][0] = value

            if use_features_as_channels:
                # swap the dimensions
                for i in range(len(self.input_data)):
                    self.input_data[i] = self.input_data[i].permute(2,1,0)


            # Determine the output data
            if task == 'regression':
                self.num_classes = 1
                if type(self.graph_labels[0]) == list:
                    self.num_classes = len(self.graph_labels[0])
            else:
                try:
                    self.num_classes = len(set(self.graph_labels))
                except:
                    self.num_classes = len(self.graph_labels[0])

            self.output_data = torch.zeros(self.num_graphs, self.num_classes)

            if task == 'regression':
                self.output_data = torch.tensor(self.graph_labels)
                self.output_data = self.output_data.unsqueeze(1)
                if output_features.get('transformation', None) is not None:
                    self.output_data = transform_data(self.output_data, output_features)

                self.output_feature_dimensions = 1
            else:
                for i, label in enumerate(self.graph_labels):
                    if type(label) == int:
                        self.output_data[i][label] = 1
                    elif type(label) == list:
                        self.output_data[i] = torch.tensor(label)
                # the output feature dimension
                self.output_feature_dimensions = self.output_data.shape[1]
            # the input channel dimension
            self.input_channels = self.input_data[0].shape[0]
            # the input feature dimension
            self.input_feature_dimensions = self.input_data[0].shape[2]
        return None

    def set_precision(self, precision:str='double'):
        """
        Adapt the precision of the input data
        :param precision: str - precision of the input data (double or float)
        """
        if precision == 'double':
            for i in range(len(self.input_data)):
                self.input_data[i] = self.input_data[i].double()
            self.output_data = self.output_data.double()
        elif precision == 'float':
            for i in range(len(self.input_data)):
                self.input_data[i] = self.input_data[i].float()
            self.output_data = self.output_data.float()


class GraphDataUnion:
    def __init__(self, db_names, graph_data):
        self.graph_db_names = db_names
        self.graph_name_to_index = {}

        # merge all the graph data into one
        self.graph_data = GraphData()
        start_index = 0
        for i, graph in enumerate(graph_data):
            if i == 0:
                self.graph_data = graph
            else:
                self.graph_data += graph
            indices = np.arange(start_index, start_index + len(graph))
            start_index += len(graph)
            self.graph_name_to_index[graph.name] = indices






        self.graph_data = graph_data


def get_graph_data(db_name: str, data_path : Path, task='graph_classification', input_features=None, output_features=None, graph_format='NEL', only_graphs=False, precision='double', experiment_config=None):
    """
    Load the graph data by name.
    :param db_name: str - name of the graph database
    :param data_path: Path - path to the data
    :param task: str - task to perform on the data
    :param input_features: dict - input features
    :param output_features: dict - output features
    :param graph_format: str - format of the data NEL: node edge label format
    :param only_graphs: bool - whether to load only the graphs

    """
    # load the graph data
    if graph_format == 'NEL':
        graph_data = GraphData()
        graph_data.load_nel_graphs(db_name=db_name, path=data_path, input_features=input_features, output_features=output_features, task=task, only_graphs=only_graphs)
    elif graph_format == 'RuleGNNDataset':
        graph_data = ShareGNNDataset(root=str(data_path),
                                     name=db_name,
                                     precision=precision,
                                     input_features=input_features,
                                     output_features=output_features,
                                     task=task,
                                     experiment_config=experiment_config)
        pass
    else:
        raise ValueError(f'Graph format {graph_format} not supported')
    return graph_data


class BenchmarkDatasets(InMemoryDataset):
    def __init__(self, root: str, name: str, graph_data: GraphData):
        self.graph_data = graph_data
        self.name = name
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=True)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.name}_Edges.txt', f'{self.name}_Nodes.txt', f'{self.name}_Labels.txt']

    @property
    def processed_file_names(self):
        return [f'data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        num_node_labels = self.graph_data.node_labels['primary'].num_unique_node_labels
        for i, graph in enumerate(self.graph_data.graphs):
            data = torch_geometric.data.Data()
            data_x = torch.zeros((graph.number_of_nodes(), num_node_labels))
            # create one hot encoding for node labels
            for j, node in graph.nodes(data=True):
                data_x[j][node['primary_node_labels']] = 1
            data.x = data_x
            edge_index = torch.zeros((2, 2 * len(graph.edges)), dtype=torch.long)
            # add each edge twice, once in each direction
            for j, edge in enumerate(graph.edges):
                edge_index[0][2 * j] = edge[0]
                edge_index[1][2 * j] = edge[1]
                edge_index[0][2 * j + 1] = edge[1]
                edge_index[1][2 * j + 1] = edge[0]

            data.edge_index = edge_index
            data.y = torch.tensor(self.graph_data.graph_labels[i])
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def zinc_to_graph_data(train, validation, test, graph_db_name, use_features=True):
    graphs = GraphData()
    graphs.name = graph_db_name
    graphs.edge_labels['primary'] = EdgeLabels()
    graphs.node_labels['primary'] = NodeLabels()
    graphs.node_labels['primary'].node_labels = []
    graphs.edge_labels['primary'].edge_labels = []
    graphs.graph_labels = []
    graphs.output_data = []
    graphs.max_nodes = 0
    graphs.num_classes = 1
    graphs.num_graphs = len(train) + len(validation) + len(test)

    max_label = 0
    label_set = set()

    original_source = -1
    for data in [train, validation, test]:
        for i, graph in enumerate(data):
            # add new graph
            graphs.graphs.append(nx.Graph())
            # add new nodes

            #graphs.edge_labels['primary'].edge_labels.append([])
            graphs.input_data.append(torch.ones(graph['x'].shape[0]).float())
            # add graph inputs using the values from graph['x'] and flatten the tensor
            if use_features:
                graphs.input_data[-1] = graph['x'].flatten().float()

            edges = graph['edge_index']
            # format edges to list of tuples
            edges = edges.T.tolist()
            # add edges to graph
            for i, edge in enumerate(edges):
                if edge[0] < edge[1]:
                    edge_label = graph['edge_attr'][i].item()
                    graphs.graphs[-1].add_edge(edge[0], edge[1], label=edge_label)
                    #graphs.edge_labels['primary'].edge_labels[-1].append(edge_label)
            # add node labels
            graphs.node_labels['primary'].node_labels.append([x.item() for x in graph['x']])
            # add also node labels to the existing graph node
            for node in graphs.graphs[-1].nodes(data=True):
                node[1]['primary_node_labels'] = graph['x'][node[0]].item()

            # update max_label
            max_label = max(abs(max_label), max(abs(graph['x'])).item())
            # add graph label
            for node_label in graph['x']:
                label_set.add(node_label.item())

            graphs.edge_labels['primary'].edge_labels.append(graph['edge_attr'])
            graphs.graph_labels.append(graph['y'].item())
            graphs.output_data.append(graph['y'].float())
            graphs.max_nodes = max(graphs.max_nodes, len(graph['x']))

            pass
        pass
    if use_features:
        # normalize graph inputs
        number_of_node_labels = len(label_set)
        label_set = sorted(label_set)
        step = 1.0 / number_of_node_labels
        for i, graph in enumerate(graphs.input_data):
            for j, val in enumerate(graph):
                graphs.input_data[i][j] = (label_set.index(val) + 1) * step * (-1) ** label_set.index(val)

    # convert one hot label list to tensor
    graphs.output_data = torch.stack(graphs.output_data)
    return graphs
