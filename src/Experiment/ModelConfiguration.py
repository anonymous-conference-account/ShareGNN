import datetime
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from src.Architectures.ShareGNN import ShareGNN, Parameters
from src.Experiment.data_sampling import curriculum_sampling
from src.Preprocessing.GraphData import GraphData
from src.Preprocessing.GraphData.GraphData import ShareGNNDataset
from src.Time.TimeClass import TimeClass
from src.utils.utils import get_k_lowest_nonzero_indices, valid_pruning_configuration, is_pruning

class EvaluationValues:
    def __init__(self):
        self.accuracy = 0.0
        self.accuracy_std = 0.0
        self.accuracy_roc_auc = 0.0
        self.loss = 0.0
        self.loss_std = 0.0
        self.mae = 0.0
        self.mae_std = 0.0



class ModelConfiguration:
    """
    This class defines a specific configuration of a model and a specific dataset for a specific run
    parameters:
    run_id: int -> id of the run
    k_val: int -> id of the k-fold validation
    graph_data: RuleGNNDataset -> the dataset including input features, labels and output features
    model_data: Tuple[np.ndarray, np.ndarray, np.ndarray] -> the training, validation and test data given as numpy arrays
    seed: int -> seed for the run, it is used to initialize the network's weights
    para: Parameters -> the parameters for the run
    """
    def __init__(self, run_id: int, k_val: int, graph_data: ShareGNNDataset, model_data: Tuple[np.ndarray, np.ndarray, np.ndarray], seed: int, para: Parameters.Parameters):
        self.num_epoch_samples = None
        self.best_epoch = None
        self.device = None
        self.dtype = None
        self.run_id = run_id
        self.k_val = k_val
        self.graph_data = graph_data
        self.training_data, self.validate_data, self.test_data = model_data
        self.seed = seed
        self.para = para
        self.results_path = para.run_config.config['paths']['results']
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.net = None
        self.class_weights = None
        # get gpu or cpu: (cpu is recommended at the moment)
        if self.para.run_config.config.get('device', None) is not None:
            self.device = torch.device(self.para.run_config.config['device'] if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float
        if self.para.run_config.config.get('precision', 'float') == 'double':
            self.dtype = torch.double

    def Run(self, pretrained_network=None):
        """
        Sets up the model and runs the training
        parameters:
        run_seed: int -> seed for the run
        net: -> optional pretrained network, if None a new network is created based on the given parameters
        """

        # Initialize the graph neural network
        self.initialize_model(pretrained_network=pretrained_network, use_model=self.para.run_config.config.get('use_model', 'ShareGNN'))
        # start the timer
        timer = TimeClass()
        # Set up the loss function
        self.set_loss_function()
        # Set up the optimizer
        self.set_optimizer()
        # Preprocess the results writer
        if not self.preprocess_writer():
            # Run already exists, so we do not run the training again
            print(f"Run {self.run_id} already exists, skipping training.")
            return
        # set the scheduler
        self.set_scheduler()

        # Store the best epoch
        self.best_epoch = {"epoch": 0, "acc": 0.0, "roc_auc": 0.0, "loss": 1000000.0, "val_acc": 0.0,  "val_roc_auc": 0.0, "val_loss": 1000000.0, "val_mae": 1000000.0}

        """
        Run through the defined number of epochs
        """
        seeds = np.arange(self.para.n_epochs*self.para.n_val_runs)
        seeds = np.reshape(seeds, (self.para.n_epochs, self.para.n_val_runs))

        # set data to device
        #self.graph_data.to(self.device)

        # Run through the epochs
        for epoch in range(self.para.n_epochs):
            # Test early stopping criterion
            if self.early_stopping(epoch):
                break

            timer.measure("epoch")
            self.net.epoch = epoch
            epoch_values = EvaluationValues()
            validation_values = EvaluationValues()
            test_values = EvaluationValues()


            # divide the whole training data into batches
            if self.para.run_config.config.get('training_data_sampling', None) is None or self.para.run_config.config['training_data_sampling'].get('type', None) == 'default':
                shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
                np.random.seed(shuffling_seed)
                np.random.shuffle(self.training_data)
                self.para.run_config.batch_size = min(self.para.run_config.batch_size, len(self.training_data))
                train_batches = np.array_split(self.training_data, self.training_data.size // self.para.run_config.batch_size)

            # sample the batches from the training data uniformly
            elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'random':
                shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
                np.random.seed(shuffling_seed)
                np.random.shuffle(self.training_data)
                self.para.run_config.batch_size = min(self.para.run_config.batch_size, len(self.training_data))
                # get random indices from the training data
                random_indices = np.random.choice(len(self.training_data), len(self.training_data), replace=True)
                train_batches = np.array_split(self.training_data[random_indices], self.training_data.size // self.para.run_config.batch_size)

            # sample the batches from the training data respecting the output class distribution
            elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'balanced':
                balancing_factor = self.para.run_config.config['training_data_sampling'].get('factor', 0.5)
                total_samples_per_epoch = self.para.run_config.config['training_data_sampling'].get('total_samples_per_epoch', 1)
                # get the class distribution of the training data
                unique_classes, class_indices, class_counts = torch.unique(self.graph_data.y[self.training_data], return_counts=True, return_inverse=True)
                indices_per_class = []
                for i in unique_classes:
                    indices_per_class.append(np.where(class_indices == i)[0])
                random_indices_per_class = []
                balancing = [1-balancing_factor, balancing_factor]
                for i in range(len(indices_per_class)):
                    random_indices_per_class.append(np.random.choice(self.training_data[indices_per_class[i]], int(total_samples_per_epoch*self.training_data.size * balancing[i]), replace=True))
                # concatenate the random indices
                random_indices = np.concatenate(random_indices_per_class)
                shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
                np.random.seed(shuffling_seed)
                np.random.shuffle(random_indices)
                train_batches = np.array_split(random_indices, self.training_data.size // self.para.run_config.batch_size)



            # sort the graphs by the number of nodes
            elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'curriculum':
                train_batches = curriculum_sampling(graph_data=self.graph_data,
                                                               training_data=self.training_data,
                                                               num_batches=self.para.run_config.config['training_data_sampling'].get('num_batches', (len(self.training_data) - 1) // self.para.run_config.batch_size + 1),
                                                               batch_size=self.para.run_config.batch_size,
                                                               bucket_num=self.para.run_config.config['training_data_sampling']['bucket_num'],
                                                               total_epochs=self.para.n_epochs,
                                                               epoch=epoch,
                                                               anti=self.para.run_config.config['training_data_sampling'].get('anti', False),
                                                               exclusive=self.para.run_config.config['training_data_sampling'].get('exclusive', True))

            # sort the graphs by the number of edges
            elif self.para.run_config.config['training_data_sampling'].get('type', None) == 'curriculum_edges':
                train_batches = curriculum_sampling(graph_data=self.graph_data,
                                                             training_data=self.training_data,
                                                             num_batches=self.para.run_config.config['training_data_sampling'].get('num_batches', (len(self.training_data) - 1) // self.para.run_config.batch_size + 1),
                                                             batch_size=self.para.run_config.batch_size,
                                                                bucket_num=self.para.run_config.config['training_data_sampling']['bucket_num'],
                                                                total_epochs=self.para.n_epochs,
                                                                epoch=epoch,
                                                                anti=self.para.run_config.config['training_data_sampling'].get('anti', False),
                                                                exclusive=self.para.run_config.config['training_data_sampling'].get('exclusive', True),
                                                                use_edges=True)
            else:
                shuffling_seed = seeds[epoch][self.k_val] * self.run_id + self.seed
                np.random.seed(shuffling_seed)
                np.random.shuffle(self.training_data)
                self.para.run_config.batch_size = min(self.para.run_config.batch_size, len(self.training_data))
                train_batches = np.array_split(self.training_data,
                                               self.training_data.size // self.para.run_config.batch_size)


            # if weighted_loss is set to true, get the class weights
            if self.para.run_config.config.get('weighted_loss', False):
                # get class counts per batch
                self.class_weights = torch.zeros((len(train_batches), self.graph_data.num_classes), dtype=self.dtype)
                for i in range(len(train_batches)):
                    self.class_weights[i] = torch.unique(self.graph_data.y[train_batches[i]], return_counts=True)[1]
                self.class_weights = 1.0 - torch.einsum('ij,i->ij', self.class_weights, 1.0/self.class_weights.sum(dim=1))



            self.num_epoch_samples = sum([batch.size for batch in train_batches])
            random_variation_bool = self.para.run_config.config.get('input_features', None).get('random_variation', None)
            self.net.train(True)
            if self.para.run_config.config['task'] in ['graph_regression', 'graph_classification']:
                self.train_graph_task(epoch=epoch, values=(epoch_values, validation_values, test_values), train_batches=train_batches, random_variation_bool=random_variation_bool, timer=timer)
            elif self.para.run_config.config['task'] == 'node_classification':
                self.train_node_task(epoch=epoch, values=(epoch_values, validation_values, test_values), train_batches=train_batches, random_variation_bool=random_variation_bool, timer=timer)


            # Pruning
            if valid_pruning_configuration(self.para, epoch):
                self.model_pruning(epoch)


            epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch,train_values=epoch_values, validation_values=validation_values, test_values=test_values, evaluation_type='validation')
            epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch,train_values=epoch_values, validation_values=validation_values, test_values=test_values, evaluation_type='test')

            timer.measure("epoch")
            epoch_time = timer.get_flag_time("epoch")

            self.postprocess_writer(epoch, epoch_time, epoch_values, validation_values, test_values)


            # apply scheduler
            if self.scheduler is not None:
                if self.para.run_config.config['scheduler']['type'] == 'ReduceLROnPlateau':
                    self.scheduler.step(validation_values.loss)
                else:
                    self.scheduler.step()

    def initialize_model(self, pretrained_network,  use_model='ShareGNN'):
        """
        Initialize the network, i.e., if pretrained_network is given load the network from the file, else create a new network
        """
        print(f'Initializing network with seed {self.seed}')
        if use_model == 'GCN':
            self.net = GCNGraph(graph_data=self.graph_data, para=self.para, seed=self.seed, device=self.device)
        else:
            if pretrained_network is not None:
                self.net = pretrained_network
            else:
                self.net = ShareGNN.ShareGNN(graph_data=self.graph_data,
                                         para=self.para,
                                         seed=self.seed, device=self.device)


        # set the network to device
        self.net.to(self.device)
        print(f'Network initialized with seed {self.seed}')

    def set_loss_function(self, *args, **kwargs):
        if self.para.run_config.loss == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss(*args, **kwargs)
        elif self.para.run_config.loss in ['MeanSquaredError', 'MSELoss', 'mse', 'MSE']:
            self.criterion = nn.MSELoss(*args, **kwargs)
        elif self.para.run_config.loss in ['RootedMeanSquaredError', 'RMSELoss', 'rmse', 'RMSE']:
            self.criterion = nn.MSELoss(*args, **kwargs)
        elif self.para.run_config.loss in ['L1Loss', 'l1', 'L1', 'mean_absolute_error', 'mae', 'MAE', 'MeanAbsoluteError']:
            self.criterion = nn.L1Loss(*args, **kwargs)
        elif self.para.run_config.loss in ['BCELoss', 'bce', 'BCE']:
            self.criterion = nn.BCELoss(*args, **kwargs)
        elif self.para.run_config.loss in ['BCEWithLogitsLoss', 'bce_with_logits', 'BCEWithLogits']:
            self.criterion = nn.BCEWithLogitsLoss(*args, **kwargs)
        elif self.para.run_config.loss in ['NLLLoss', 'nll', 'NLL']:
            self.criterion = nn.NLLLoss(*args, **kwargs)
        else:
            raise ValueError(f"Loss function {self.para.run_config.loss} not implemented")

    def set_optimizer(self):
        if self.para.run_config.optimizer == 'Adam':
            opt = optim.Adam
        elif self.para.run_config.optimizer == 'AdamW':
            opt = optim.AdamW
        elif self.para.run_config.optimizer == 'SGD':
            opt = optim.SGD
        elif self.para.run_config.optimizer == 'RMSprop':
            opt = optim.RMSprop
        elif self.para.run_config.optimizer == 'Adadelta':
            opt = optim.Adadelta
        elif self.para.run_config.optimizer == 'Adagrad':
            opt = optim.Adagrad
        else:
            opt = optim.Adam

        self.optimizer = opt(self.net.parameters(), lr=self.para.learning_rate, weight_decay=self.para.run_config.weight_decay)

    def set_scheduler(self):
        """
        Variable learning rate
        """
        if self.para.run_config.config.get('scheduler', None) is not None:
            scheduler = self.para.run_config.config.get('scheduler')
            if isinstance(scheduler, bool):
                if scheduler is True:
                    raise ValueError("Scheduler is set to True, but no scheduler is defined")
                else:
                    self.scheduler = None
                    return
            scheduler_type = scheduler.get('type', None)
            if scheduler_type == 'StepLR':
                self.scheduler = StepLR(self.optimizer, step_size=scheduler.get('step_size', None), gamma=scheduler.get('gamma', None))
            elif scheduler_type == 'ReduceLROnPlateau':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=scheduler.get('patience', 10), min_lr=scheduler.get('min_lr', 0), factor=scheduler.get('factor', 0.1))


    def early_stopping(self, epoch):
        if self.para.run_config.config.get('early_stopping', {'enabled': False})['enabled']:
            if epoch - self.best_epoch["epoch"] > self.para.run_config.config['early_stopping']['patience']:
                if self.para.print_results:
                    print(f"Early stopping at epoch {epoch}")
                return True
        return False

    def test_weight_update(self, weights):
        weight_changes = []
        for i, layer in enumerate(self.net.net_layers):
            change = np.array(
                [weights[i][j] - x.item() for j, x in enumerate(layer.Param_W)]).flatten().reshape(1, -1)
            weight_changes.append(change)
            # save to three differen csv files using pandas
            # df = pd.DataFrame(change)
            # df.to_csv(f'Results/Parameter/layer_{i}_change.csv', header=False, index=False, mode='a')
            # if there is some change print that the layer trains
            if np.count_nonzero(change) > 0:
                print(f'Layer {i} has updated')
            else:
                print(f'Layer {i} has not updated')

    def model_pruning(self, epoch):
        # prune each five epochs

        print('Pruning')
        # iterate over the layers of the neural net
        for i, layer in enumerate(self.net.net_layers):
            pruning_per_layer = self.para.run_config.config['prune']['percentage'][i]
            # use total number of epochs, the epoch step and the pruning percentage
            pruning_per_layer /= (self.para.n_epochs / self.para.run_config.config['prune']['epochs']) - 1

            # get tensor from the parameter_list layer.Param_W
            layer_tensor = torch.abs(torch.tensor(layer.Param_W) * torch.tensor(layer.mask))
            # print number of non zero entries in layer_tensor
            print(f'Number of non zero entries in before pruning {layer.name}: {torch.count_nonzero(layer_tensor)}')
            # get the indices of the trainable parameters with lowest absolute max(1, 1%)
            k = int(layer_tensor.size(0) * pruning_per_layer)
            if k != 0:
                low = torch.topk(layer_tensor, k, largest=False)
                lowest_indices = get_k_lowest_nonzero_indices(layer_tensor, k)
                # set all indices in layer.mask to zero
                layer.mask[lowest_indices] = 0
                layer.Param_W.data = layer.Param_W_original * layer.mask
                # for c, graph_weight_distribution in enumerate(layer.weight_distribution):
                #     new_graph_weight_distribution = None
                #     for [i, j, pos] in graph_weight_distribution:
                #         # if pos is in lowest_indices do nothing else append to new_graph_weight_distribution
                #         if pos in lowest_indices:
                #             pass
                #         else:
                #             if new_graph_weight_distribution is None:
                #                 new_graph_weight_distribution = np.array([i, j, pos])
                #             else:
                #                 # add [i, j, pos] to new_graph_weight_distribution
                #                 new_graph_weight_distribution = np.vstack((new_graph_weight_distribution, [i, j, pos]))
                #     layer.weight_distribution[c] = new_graph_weight_distribution

            # print number of non zero entries in layer.Param_W
            print(
                f'Number of non zero entries in layer after pruning {layer.name}: {torch.count_nonzero(layer.Param_W)}')
        if is_pruning(self.para.run_config.config):
            for i, layer in enumerate(self.net.net_layers):
                # get tensor from the parameter_list layer.Param_W
                layer_tensor = torch.abs(torch.tensor(layer.Param_W).clone().detach() * torch.tensor(layer.mask))
                # print number of non zero entries in layer_tensor
                print(
                    f'Number of non zero entries in layer {layer.name}: {torch.count_nonzero(layer_tensor)}/{torch.numel(layer_tensor)}')

                # multiply the Param_W with the mask
                layer.Param_W.data = layer.Param_W.data * layer.mask


    def evaluate_results(self, epoch: int,
                         train_values: EvaluationValues,
                         validation_values: EvaluationValues,
                         test_values: EvaluationValues,
                         evaluation_type,
                         outputs=None,
                         labels=None,
                         batch_idx=0,
                         batch_length=0,
                         num_batches=0,
                         batches=None):
        if evaluation_type == 'training':
            batch_acc = 0

            # if num classes is one calculate the mae and mae_std or if the task is regression
            if self.para.run_config.task == 'graph_regression':
                # flatten the labels and outputs
                flatten_labels = labels.detach().clone().flatten()
                flatten_outputs = outputs.detach().clone().flatten()
                if self.para.run_config.config.get('output_features_inverse', None) is not None:
                    flatten_labels = GraphData.transform_data(flatten_labels, self.para.run_config.config['output_features_inverse'])
                    flatten_outputs = GraphData.transform_data(flatten_outputs, self.para.run_config.config['output_features_inverse'])
                if self.para.run_config.config.get('invert_outputs', None) is not None:
                    if isinstance(self.para.run_config.config['invert_outputs'], dict):
                        if self.para.run_config.config['invert_outputs'].get('normalization', 'standard') == 'standard':
                            flatten_labels = flatten_labels*(self.graph_data.data['original_y'].std() + 1e-8) + self.graph_data.data['original_y'].mean()
                            flatten_outputs = flatten_outputs*(self.graph_data.data['original_y'].std() + 1e-8) + self.graph_data.data['original_y'].mean()
                        elif self.para.run_config.config['invert_outputs'].get('normalization', 'standard') == 'minmax':
                            flatten_labels = flatten_labels * (self.graph_data.data['original_y'].max() - self.graph_data.data['original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                            flatten_outputs = flatten_outputs * (self.graph_data.data['original_y'].max() - self.graph_data.data['original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                        elif self.para.run_config.config['invert_outputs'].get('normalization', 'standard') == 'minmax_zero':
                            flatten_labels = (0.5 * flatten_labels + 0.5) * (self.graph_data.data['original_y'].max() - self.graph_data.data['original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()


                batch_mae = torch.mean(torch.abs(flatten_labels - flatten_outputs))
                batch_mae_std = torch.std(torch.abs(flatten_labels - flatten_outputs))
                train_values.mae += batch_mae * (batch_length / len(self.training_data))
                train_values.mae_std += batch_mae_std * (batch_length / len(self.training_data))
            else:
                prediction = torch.argmax(outputs, dim=1)
                batch_acc = 100 * torch.sum(prediction == labels).item() / len(labels)
                train_values.accuracy += batch_acc * (batch_length / len(self.training_data))
                # roc_auc
                #batch_roc_auc = sklearn.metrics.roc_auc_score(labels, prediction)
                #train_values.accuracy_roc_auc += batch_roc_auc * (batch_length / len(self.training_data))

            if self.para.print_results:
                if self.graph_data.num_classes == 1 or self.para.run_config.task == 'graph_regression':
                    print(
                        "\tepoch: {}/{}, batch: {}/{}, loss: {}, acc: {} %, mae: {}, mae_std: {}".format(epoch + 1,
                                                                                                         self.para.n_epochs,
                                                                                                         batch_idx + 1,
                                                                                                         num_batches,
                                                                                                         train_values.loss,
                                                                                                         batch_acc,
                                                                                                         train_values.mae,
                                                                                                         train_values.mae_std))
                else:
                    print("\tepoch: {}/{}, batch: {}/{}, loss: {}, acc: {} % ".format(epoch + 1, self.para.n_epochs,
                                                                                      batch_idx + 1,
                                                                                      num_batches,
                                                                                      train_values.loss, batch_acc))
            self.para.count += 1

            if self.para.save_prediction_values:
                # print outputs and labels to a csv file
                outputs_np = outputs.detach().numpy()
                # transpose the numpy array
                outputs_np = outputs_np.T
                df = pd.DataFrame(outputs_np)
                # show only two decimal places
                df = df.round(2)
                df.to_csv("Results/Parameter/training_predictions.csv", header=False, index=False, mode='a')
                labels_np = labels.detach().numpy()
                labels_np = labels_np.T
                df = pd.DataFrame(labels_np)
                df.to_csv("Results/Parameter/training_predictions.csv", header=False, index=False, mode='a')

        elif evaluation_type == 'validation':
            '''
            Evaluate the validation accuracy for each epoch
            '''
            if self.validate_data.size != 0:
                if self.para.run_config.task in ['graph_classification', 'graph_regression']:
                    labels, outputs = self.evaluate_graph_task(self.validate_data)
                elif self.para.run_config.task == 'node_classification':
                    labels, outputs = self.evaluate_node_task(self.validate_data)
                else:
                    raise ValueError(f"Task {self.para.run_config.task} not implemented")
                # get validation loss
                validation_loss = self.criterion(outputs, labels).item()
                validation_values.loss = validation_loss
                if self.para.run_config.task == 'graph_regression':
                    flatten_labels = labels.detach().clone().flatten()
                    flatten_outputs = outputs.detach().clone().flatten()
                    if self.para.run_config.config.get('output_features_inverse', None) is not None:
                        flatten_labels = GraphData.transform_data(flatten_labels, self.para.run_config.config[
                            'output_features_inverse'])
                        flatten_outputs = GraphData.transform_data(flatten_outputs, self.para.run_config.config[
                            'output_features_inverse'])
                    if self.para.run_config.config.get('invert_outputs', None) is not None:
                        if isinstance(self.para.run_config.config['invert_outputs'], dict):
                            if self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                 'standard') == 'standard':
                                flatten_labels = flatten_labels * (self.graph_data.data['original_y'].std() + 1e-8) + \
                                                 self.graph_data.data['original_y'].mean()
                                flatten_outputs = flatten_outputs * (self.graph_data.data['original_y'].std() + 1e-8) + \
                                                  self.graph_data.data['original_y'].mean()
                            elif self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                   'standard') == 'minmax':
                                flatten_labels = flatten_labels * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                                flatten_outputs = flatten_outputs * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                            elif self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                   'standard') == 'minmax_zero':
                                flatten_labels = (0.5 * flatten_labels + 0.5) * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                    validation_mae = torch.mean(torch.abs(flatten_labels - flatten_outputs))
                    validation_values.mae = validation_mae
                    validation_mae_std = torch.std(torch.abs(flatten_labels - flatten_outputs))
                    validation_values.mae_std = validation_mae_std
                else:
                    prediction = torch.argmax(outputs, dim=1)
                    validation_acc = 100 * torch.sum(prediction==labels).item() / len(labels)
                    validation_values.accuracy = validation_acc
                    if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                        # roc_auc
                        validation_roc_auc = sklearn.metrics.roc_auc_score(labels, prediction)
                        validation_values.accuracy_roc_auc = validation_roc_auc

                # update best epoch
                if self.para.run_config.task == 'graph_regression':
                    if validation_values.mae <= self.best_epoch["val_mae"] or valid_pruning_configuration(self.para, epoch):
                        self.best_epoch["epoch"] = epoch
                        self.best_epoch["acc"] = train_values.accuracy
                        self.best_epoch["roc_auc"] = train_values.accuracy_roc_auc
                        self.best_epoch["loss"] = train_values.loss
                        self.best_epoch["val_acc"] = validation_values.accuracy
                        self.best_epoch["val_roc_auc"] = validation_values.accuracy_roc_auc
                        self.best_epoch["val_loss"] = validation_values.loss
                        self.best_epoch["val_mae"] = validation_values.mae
                        self.best_epoch["val_mae_std"] = validation_values.mae_std
                        # save the best model
                        best_model_path = self.results_path.joinpath(f'{self.para.db}/Models/')
                        if not os.path.exists(best_model_path):
                            os.makedirs(best_model_path)
                        # Save the model if best model is used
                        if 'best_model' in self.para.run_config.config and self.para.run_config.config['best_model']:
                            final_path = self.results_path.joinpath(f'{self.para.db}/Models/model_{self.para.config_id}_run_{self.run_id}_val_step_{self.k_val}.pt')
                            torch.save(self.net.state_dict(),final_path)


                else:
                    acc_condition = (validation_values.accuracy > self.best_epoch["val_acc"] or validation_values.accuracy == self.best_epoch[
                        "val_acc"] and validation_loss < self.best_epoch["val_loss"])
                    roc_condition = (validation_values.accuracy_roc_auc > self.best_epoch["val_roc_auc"] or validation_values.accuracy_roc_auc == self.best_epoch[
                        "val_roc_auc"] and validation_loss < self.best_epoch["val_loss"])
                    loss_condition = (validation_loss < self.best_epoch["val_loss"])
                    condition = False
                    if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'accuracy':
                        condition = acc_condition
                    elif self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                        condition = roc_condition
                    elif self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'loss':
                        condition = loss_condition
                    # check if pruning is on, then save the best model in the last pruning epoch
                    if condition or valid_pruning_configuration(self.para, epoch):
                        self.best_epoch["epoch"] = epoch
                        self.best_epoch["acc"] = train_values.accuracy
                        self.best_epoch["roc_auc"] = train_values.accuracy_roc_auc
                        self.best_epoch["loss"] = train_values.loss
                        self.best_epoch["val_acc"] = validation_values.accuracy
                        self.best_epoch["val_roc_auc"] = validation_values.accuracy_roc_auc
                        self.best_epoch["val_loss"] = validation_values.loss
                        # save the best model
                        best_model_path = self.results_path.joinpath(f'{self.para.db}/Models/')
                        if not os.path.exists(best_model_path):
                            os.makedirs(best_model_path)
                        # Save the model if best model is used
                        if self.para.run_config.config.get('best_model', False) or self.para.run_config.config.get('save_best_model', False):
                            final_path = self.results_path.joinpath(f'{self.para.db}/Models/model_{self.para.config_id}_run_{self.run_id}_val_step_{self.k_val}.pt')
                            torch.save(self.net.state_dict(), final_path)

            if self.para.save_prediction_values:
                # print outputs and labels to a csv file
                outputs_np = outputs.detach().numpy()
                # transpose the numpy array
                outputs_np = outputs_np.T
                df = pd.DataFrame(outputs_np)
                # show only two decimal places
                df = df.round(2)
                df.to_csv("Results/Parameter/validation_predictions.csv", header=False, index=False, mode='a')
                labels_np = labels.detach().numpy()
                labels_np = labels_np.T
                df = pd.DataFrame(labels_np)
                df.to_csv("Results/Parameter/validation_predictions.csv", header=False, index=False, mode='a')

        elif evaluation_type == 'test':
            # Test accuracy
            # print only if run best model is used
            if self.para.run_config.config.get('best_model', False):
                if self.para.run_config.task in ['graph_classification', 'graph_regression']:
                    labels, outputs = self.evaluate_graph_task(self.test_data)
                elif self.para.run_config.task == 'node_classification':
                    labels, outputs = self.evaluate_node_task(self.test_data)
                else:
                    raise ValueError(f"Task {self.para.run_config.task} not implemented")

                test_loss = self.criterion(outputs, labels).item()
                test_values.loss = test_loss
                if self.para.run_config.task == 'graph_regression':
                    flatten_labels = labels.detach().clone().flatten()
                    flatten_outputs = outputs.detach().clone().flatten()
                    if self.para.run_config.config.get('output_features_inverse', None) is not None:
                        flatten_labels = GraphData.transform_data(flatten_labels, self.para.run_config.config[
                            'output_features_inverse'])
                        flatten_outputs = GraphData.transform_data(flatten_outputs, self.para.run_config.config[
                            'output_features_inverse'])
                    if self.para.run_config.config.get('invert_outputs', None) is not None:
                        if isinstance(self.para.run_config.config['invert_outputs'], dict):
                            if self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                 'standard') == 'standard':
                                flatten_labels = flatten_labels * (self.graph_data.data['original_y'].std() + 1e-8) + \
                                                 self.graph_data.data['original_y'].mean()
                                flatten_outputs = flatten_outputs * (self.graph_data.data['original_y'].std() + 1e-8) + \
                                                  self.graph_data.data['original_y'].mean()
                            elif self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                   'standard') == 'minmax':
                                flatten_labels = flatten_labels * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                                flatten_outputs = flatten_outputs * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                            elif self.para.run_config.config['invert_outputs'].get('normalization',
                                                                                   'standard') == 'minmax_zero':
                                flatten_labels = (0.5 * flatten_labels + 0.5) * (
                                            self.graph_data.data['original_y'].max() - self.graph_data.data[
                                        'original_y'].min() + 1e-8) + self.graph_data.data['original_y'].min()
                    test_mae = torch.mean(torch.abs(flatten_labels - flatten_outputs))
                    test_values.mae = test_mae
                    test_mae_std = torch.std(torch.abs(flatten_labels - flatten_outputs))
                    test_values.mae_std = test_mae_std
                else:
                    prediction = torch.argmax(outputs, dim=1)
                    test_acc = 100 * torch.sum(prediction == labels).item() / len(labels)
                    test_values.accuracy = test_acc
                    if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                        # roc_auc
                        test_roc_auc = sklearn.metrics.roc_auc_score(labels, prediction)
                        test_values.accuracy_roc_auc = test_roc_auc

                if self.para.print_results:
                    np_labels = labels.detach().numpy()
                    np_outputs = outputs.detach().numpy()
                    # np array of correct/incorrect predictions
                    labels_argmax = np_labels.argmax(axis=1)
                    outputs_argmax = np_outputs.argmax(axis=1)
                    # change if task is graph_regression
                    if 'task' in self.para.run_config.config and self.para.run_config.config['task'] == 'graph_regression':
                        np_correct = np_labels - np_outputs
                    else:
                        np_correct = labels_argmax == outputs_argmax
                    # print entries of np_labels and np_outputs
                    for j, data_pos in enumerate(self.test_data, 0):
                        print(data_pos, np_labels[j], np_outputs[j], np_correct[j])

                if self.para.save_prediction_values:
                    # print outputs and labels to a csv file
                    outputs_np = outputs.detach().numpy()
                    # transpose the numpy array
                    outputs_np = outputs_np.T
                    df = pd.DataFrame(outputs_np)
                    # show only two decimal places
                    df = df.round(2)
                    df.to_csv("Results/Parameter/test_predictions.csv", header=False, index=False, mode='a')
                    labels_np = labels.detach().numpy()
                    labels_np = labels_np.T
                    df = pd.DataFrame(labels_np)
                    df.to_csv("Results/Parameter/test_predictions.csv", header=False, index=False, mode='a')

        return train_values, validation_values, test_values

    def preprocess_writer(self)-> bool:
        if self.run_id == 0 and self.k_val == 0:
            # create a file about the net details including (net, optimizer, learning rate, loss function, batch size, number of classes, number of epochs, balanced data, dropout)
            file_name = f'{self.para.db}_{self.para.config_id}_Network.txt'
            final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
            with open(final_path, "a") as file_obj:
                file_obj.write(f"Network architecture: {self.para.run_config.network_architecture}\n"
                               f"Optimizer: {self.optimizer}\n"
                               f"Loss function: {self.criterion}\n"
                               f"Batch size: {self.para.batch_size}\n"
                               f"Balanced data: {self.para.balance_data}\n"
                               f"Number of epochs: {self.para.n_epochs}\n")
                # iterate over the layers of the neural net
                total_trainable_parameters = 0
                for layer in self.net.net_layers:
                    file_obj.write(f"\n")
                    try:
                        file_obj.write(f"Layer: {layer.name}\n")
                    except:
                        file_obj.write(f"Linear Layer\n")
                    file_obj.write(f"\n")
                    # get number of trainable parameters
                    layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    total_trainable_parameters += layer_params
                    file_obj.write(f"Layer Trainable Parameters: {layer_params}\n")
                    try:
                        file_obj.write(f"Node labels: {layer.node_labels.num_unique_node_labels}\n")
                    except:
                        pass
                    try:
                        for i, n in enumerate(layer.n_properties):
                            file_obj.write(f"Number of Source Labels (type: {layer.source_label_descriptions[i]}) in channel {i}: {layer.n_source_labels[i]}\n")
                            file_obj.write(f"Number of Target Labels (type: {layer.target_label_descriptions[i]}) in channel {i}: {layer.n_target_labels[i]}\n")
                            if layer.bias_list[i]:
                                file_obj.write(f"Number of Bias Labels in channel {i}: {layer.n_bias_labels[i]}\n")
                            file_obj.write(f"Number of pairwise properties in channel {i}: {n}\n")
                            file_obj.write("\n")
                    except:
                        pass
                    try:
                        for i, n in enumerate(layer.n_node_labels):
                            file_obj.write(f"Number of Node Labels (type: {layer.node_label_descriptions[i]}) in channel {i}: {n}\n")
                            file_obj.write("\n")
                    except:
                        pass

                    weight_learnable_parameters = 0
                    bias_learnable_parameters = 0
                    try:
                        if layer.Param_W.requires_grad:
                            weight_learnable_parameters += layer.Param_W.numel()
                    except:
                        pass
                    try:
                        if layer.Param_b.requires_grad:
                            bias_learnable_parameters += layer.Param_b.numel()
                    except:
                        pass

                    file_obj.write("Weight matrix learnable parameters: {}\n".format(weight_learnable_parameters))
                    file_obj.write("Bias learnable parameters: {}\n".format(bias_learnable_parameters))
                    try:
                        file_obj.write(f"Edge labels: {layer.edge_labels.num_unique_edge_labels}\n")
                    except:
                        pass
                for name, param in self.net.named_parameters():
                    file_obj.write(f"Layer: {name} -> {param.requires_grad}\n")

                file_obj.write(f"\n")
                file_obj.write(f"Total trainable parameters: {total_trainable_parameters}\n")

        file_name = f'{self.para.db}_{self.para.config_id}_Results_run_id_{self.run_id}_validation_step_{self.para.validation_id}.csv'

        # if the file does not exist create a new file
        with open(self.results_path.joinpath(f'{self.para.db}/Results/{file_name}'), "w") as file_obj:
            file_obj.write("")

        # header use semicolon as delimiter
        if self.para.run_config.task == 'graph_regression':
            header = "Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss;" \
                     "EpochAccuracy;EpochTime;EpochMAE;EpochMAEStd;ValidationLoss;ValidationAccuracy;ValidationMAE;ValidationMAEStd;TestLoss;TestAccuracy;TestMAE;TestMAEStd\n"
        else:
            if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                header = "Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss;" \
                         "EpochAccuracy;EpochAUC;EpochTime;ValidationAccuracy;ValidationLoss;ValidationAUC;TestAccuracy;TestLoss;TestAUC\n"
            else:
                header = "Dataset;Time;RunNumber;ValidationNumber;Seed;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss;EpochAccuracy;" \
                         "EpochTime;ValidationAccuracy;ValidationLoss;TestAccuracy;TestLoss\n"

        # Save file for results and add header if the file is new
        final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
        with open(final_path, "a") as file_obj:
            if os.stat(final_path).st_size == 0:
                file_obj.write(header)
        return True


    def postprocess_writer(self, epoch, epoch_time, train_values: EvaluationValues, validation_values: EvaluationValues, test_values: EvaluationValues):
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.para.print_results:
            # if class num is one print the mae and mse
            if self.para.run_config.task == 'graph_regression':
                print(
                    f'run: {self.run_id} val step: {self.k_val} epoch: {epoch + 1}/{self.para.n_epochs} epoch loss: {train_values.loss} epoch acc: {train_values.accuracy} epoch mae: {train_values.mae} +- {train_values.mae_std} epoch time: {epoch_time}'
                    f' validation acc: {validation_values.accuracy} validation loss: {validation_values.loss} validation mae: {validation_values.mae} +- {validation_values.mae_std}'
                    f'test acc: {test_values.accuracy} test loss: {test_values.loss} test mae: {test_values.mae} +- {test_values.mae_std}'
                    f'time: {epoch_time}')
            else:
                print(
                    f'run: {self.run_id} val step: {self.k_val} epoch: {epoch + 1}/{self.para.n_epochs} epoch loss: {train_values.loss} epoch acc: {train_values.accuracy}'
                    f' validation acc: {validation_values.accuracy} validation loss: {validation_values.loss}'
                    f'test acc: {test_values.accuracy} test loss: {test_values.loss}'
                    f'time: {epoch_time}')

        if self.para.run_config.task == 'graph_regression':
            res_str =   f"{self.para.db};{time};{self.run_id};{self.k_val};{self.seed};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                        f"{train_values.loss};{train_values.accuracy};{epoch_time};{train_values.mae};{train_values.mae_std};" \
                        f"{validation_values.loss};{validation_values.accuracy};{validation_values.mae};{validation_values.mae_std};" \
                        f"{test_values.loss};{test_values.accuracy};{test_values.mae};{test_values.mae_std}\n"
        else:
            if self.para.run_config.config.get('evaluation_metric', 'accuracy') == 'roc_auc':
                res_str =   f"{self.para.db};{time};{self.run_id};{self.k_val};{self.seed};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                            f"{train_values.loss};{train_values.accuracy};{train_values.accuracy_roc_auc};{epoch_time};" \
                            f"{validation_values.accuracy};{validation_values.loss};{validation_values.accuracy_roc_auc};" \
                            f"{test_values.accuracy};{test_values.loss};{test_values.accuracy_roc_auc}\n"
            else:
                res_str =   f"{self.para.db};{time};{self.run_id};{self.k_val};{self.seed};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                            f"{train_values.loss};{train_values.accuracy};{epoch_time};" \
                            f"{validation_values.accuracy};{validation_values.loss};" \
                            f"{test_values.accuracy};{test_values.loss}\n"

        # Save file for results
        file_name = f'{self.para.db}_{self.para.config_id}_Results_run_id_{self.run_id}_validation_step_{self.para.validation_id}.csv'
        final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
        with open(final_path, "a") as file_obj:
            file_obj.write(res_str)



    def train_graph_task(self, epoch, values, train_batches, random_variation_bool, timer):
        for batch_counter, batch in enumerate(train_batches, 0):
            timer.measure("forward")
            self.optimizer.zero_grad()
            if self.graph_data.num_classes == 1:
                outputs = torch.zeros((len(batch)), dtype=self.dtype).to(self.device)
            else:
                outputs = torch.zeros((len(batch), self.graph_data.num_classes), dtype=self.dtype).to(self.device)

            # TODO batch in one matrix ?
            for j, graph_id in enumerate(batch, 0):
                timer.measure("forward_step")
                if random_variation_bool:
                    mean = self.para.run_config.config['input_features']['random_variation'].get('mean', 0.0)
                    std = self.para.run_config.config['input_features']['random_variation'].get('std', 0.1)
                    if self.para.run_config.config.get('precision', 'double') == 'float':
                        random_variation = torch.normal(mean=mean, std=std, size=self.graph_data[graph_id].x.size(),
                                                        dtype=torch.float)
                    else:
                        random_variation = torch.normal(mean=mean, std=std, size=self.graph_data[graph_id].x.size(),
                                                        dtype=torch.double)
                    outputs[j] = self.net(self.graph_data[graph_id].x + random_variation, graph_id)
                else:
                    outputs[j] = self.net(self.graph_data[graph_id].x, graph_id)
                timer.measure("forward_step")

            # calculate the loss
            if self.para.run_config.config.get('weighted_loss', False):
                self.set_loss_function(weight =self.class_weights[batch_counter])

            loss = self.criterion(outputs, self.graph_data.y[batch])
            timer.measure("forward")

            weights = []
            # save the weights to test if they are updated (only in debug mode)
            if self.para.save_weights:
                for i, layer in enumerate(self.net.net_layers):
                    weights.append([x.item() for x in layer.Param_W])

            timer.measure("backward")
            loss.backward()
            self.optimizer.step()
            timer.measure("backward")
            timer.reset()

            # test if the weights are updated (only in debug mode)
            if self.para.save_weights:
                self.test_weight_update(weights)

            epoch_values, validation_values, test_values = values
            epoch_values.loss += loss.item()

            # Get the training accuracy
            epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch, train_values=epoch_values,
                                                                                 validation_values=validation_values,
                                                                                 test_values=test_values,
                                                                                 evaluation_type='training',
                                                                                 outputs=outputs,
                                                                                 labels=self.graph_data.y[batch],
                                                                                 batch_idx=batch_counter,
                                                                                 batch_length=len(batch),
                                                                                 num_batches=len(train_batches),
                                                                                 batches=train_batches)

    def evaluate_graph_task(self, data):
        labels = self.graph_data.y[data]
        if self.graph_data.num_classes == 1:
            outputs = torch.zeros((len(data)), dtype=self.dtype).to(self.device)
        else:
            outputs = torch.zeros((len(data), self.graph_data.num_classes), dtype=self.dtype).to(
                self.device)

        # use torch no grad to save memory
        with torch.no_grad():
            for j, data_pos in enumerate(data):
                self.net.train(False)
                outputs[j] = self.net(self.graph_data[data_pos].x, data_pos)
        return labels, outputs

    def train_node_task(self, epoch, values, train_batches, random_variation_bool, timer):
        for batch_counter, batch in enumerate(train_batches, 0):
            timer.measure("forward")
            self.optimizer.zero_grad()
            timer.measure("forward_step")
            if random_variation_bool:
                mean = self.para.run_config.config['input_features']['random_variation'].get('mean', 0.0)
                std = self.para.run_config.config['input_features']['random_variation'].get('std', 0.1)
                if self.para.run_config.config.get('precision', 'double') == 'float':
                    random_variation = torch.normal(mean=mean, std=std, size=self.graph_data[0].x.size(),
                                                    dtype=torch.float)
                else:
                    random_variation = torch.normal(mean=mean, std=std, size=self.graph_data[0].x.size(),
                                                    dtype=torch.double)
                outputs = self.net(self.graph_data[0].x + random_variation, 0)
            else:
                outputs = self.net(self.graph_data[0].x, 0)
                timer.measure("forward_step")

            # calculate the loss
            # squeeze second dimension if it is one
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            loss = self.criterion(outputs[batch], self.graph_data.y[batch])
            timer.measure("forward")

            weights = []
            # save the weights to test if they are updated (only in debug mode)
            if self.para.save_weights:
                for i, layer in enumerate(self.net.net_layers):
                    weights.append([x.item() for x in layer.Param_W])

            timer.measure("backward")
            loss.backward()
            self.optimizer.step()
            timer.measure("backward")
            timer.reset()

            # test if the weights are updated (only in debug mode)
            if self.para.save_weights:
                self.test_weight_update(weights)

            epoch_values, validation_values, test_values = values
            epoch_values.loss += loss.item()

            # Get the training accuracy
            epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch, train_values=epoch_values,
                                                                                 validation_values=validation_values,
                                                                                 test_values=test_values,
                                                                                 evaluation_type='training',
                                                                                 outputs=outputs[batch],
                                                                                 labels=self.graph_data.y[batch],
                                                                                 batch_idx=batch_counter,
                                                                                 batch_length=len(batch),
                                                                                 num_batches=len(train_batches))

    def evaluate_node_task(self, data):
        labels = self.graph_data.y[data]

        # use torch no grad to save memory
        with torch.no_grad():
            self.net.train(False)
            outputs = self.net(self.graph_data[0].x, 0)
            # squeeze second dimension if it is one
            if outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
        return labels, outputs[data]



