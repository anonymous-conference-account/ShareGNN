from abc import abstractmethod

import torch
from torch import nn

from src.Preprocessing.GraphData.GraphData import ShareGNNDataset
from src.Architectures.ShareGNN.Parameters import Parameters


class BaseGNN(nn.Module):
    def __init__(self, graph_data: ShareGNNDataset, para: Parameters, seed, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_data = graph_data
        self.para = para
        self.print_weights = self.para.net_print_weights
        self.dropout = nn.Dropout(p=self.para.dropout)
        self.convolution_grad = self.para.run_config.config.get('convolution_grad', True)
        self.aggregation_grad = self.para.run_config.config.get('aggregation_grad', True)
        self.out_dim = self.graph_data.num_classes
        precision = para.run_config.config.get('precision', 'float')
        self.module_precision = torch.float
        if precision == 'double':
            self.module_precision = torch.double
        self.aggregation_out_dim = 0
        self.af = self.get_activation_function('activation')
        self.out_af = self.get_activation_function('output_activation')
        self.net_layers = nn.ModuleList()

        self.initialize_graph_neural_network()



    def get_activation_function(self, key):
        if key in self.para.run_config.config and self.para.run_config.config[key] in ['None', 'Identity', 'identity', 'Id']:
            return nn.Identity()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Relu', 'ReLU']:
            return nn.ReLU()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['LeakyRelu', 'LeakyReLU']:
            return nn.LeakyReLU()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Tanh', 'tanh']:
            return nn.Tanh()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Sigmoid', 'sigmoid']:
            return nn.Sigmoid()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Softmax', 'softmax']:
            return nn.Softmax(dim=0)
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['LogSoftmax', 'logsoftmax', 'log_softmax']:
            return nn.LogSoftmax(dim=0)
        else:
            raise ValueError(f'Activation function {key} not recognized')


    @abstractmethod
    def initialize_graph_neural_network(self):
        pass


    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
