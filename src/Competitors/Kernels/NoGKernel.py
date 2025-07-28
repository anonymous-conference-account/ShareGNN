import os
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC, SVR
import torch


class NoGKernel():
    def __init__(self, out_path:Path, graph_data, run_num:int, validation_num:int, training_data: List[int], validate_data: List[int], test_data: List[int],
                 seed: int):
        self.out_path = out_path
        self.graph_data = graph_data
        self.training_data = training_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.seed = seed
        self.run_num = run_num
        self.validation_num = validation_num

    def Run(self):
        # create numpy vector from the graph data labels
        primary_node_labels = self.graph_data.node_labels.get('primary', None)
        primary_edge_labels = self.graph_data.edge_labels.get('primary', None)
        # get unique values with inverse indices
        unique_node_labels = torch.unique(primary_node_labels, return_inverse=True)
        num_unique_node_labels = len(unique_node_labels[0])
        if primary_edge_labels is not None:
            unique_edge_labels = torch.unique(primary_edge_labels, return_inverse=True)
        else:
            unique_edge_labels = (torch.tensor([0]), torch.zeros(self.graph_data.slices['edge_index'][-1].item()))
        num_unique_edge_labels = len(unique_edge_labels[0])
        X = np.zeros(shape=(len(self.graph_data), num_unique_node_labels + num_unique_edge_labels))


        # fill the numpy vector with number of unique node and edge labels
        for i, graph in enumerate(self.graph_data):
            for node_idx in range(self.graph_data.slices['x'][i], self.graph_data.slices['x'][i + 1]):
                X[i, unique_node_labels[1][node_idx]] += 1
            for edge_idx in range(self.graph_data.slices['edge_index'][i], self.graph_data.slices['edge_index'][i + 1]):
                X[i, int(num_unique_node_labels + unique_edge_labels[1][edge_idx])] += 1

        # normalize the data
        X = X / np.sum(X, axis=1)[:, None]

        Y = np.asarray(self.graph_data.y)
        # split the data in training, validation and test set
        X_train = X[self.training_data]
        Y_train = Y[self.training_data]
        X_val = X[self.validate_data]
        Y_val = Y[self.validate_data]
        X_test = X[self.test_data]
        Y_test = Y[self.test_data]

        for c_param in [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11, 13, 15]:
            c_param = 2 ** c_param
            # create a SVM based on an RBF kernel that trains on the training data
            # and predicts the labels of the validation data and test data
            if type(Y_train) is not np.ndarray:
                clf = SVR(kernel='rbf', C=c_param)
                clf = MultiOutputRegressor(clf)
            else:
                clf = SVC(kernel='rbf', C=c_param, random_state=self.seed)
            clf.fit(X_train, Y_train)
            Y_val_pred = clf.predict(X_val)
            if len(X_test) > 0:
                Y_test_pred = clf.predict(X_test)
            if type(Y_train) is not np.ndarray:
                val_acc = mean_absolute_error(Y_val, Y_val_pred)
                if len(X_test) > 0:
                    test_acc = mean_absolute_error(Y_test, Y_test_pred)
                else:
                    test_acc = 0
            else:
                # calculate the accuracy of the prediction
                val_acc = np.mean(Y_val_pred == Y_val)
                if len(X_test) > 0:
                    test_acc = np.mean(Y_test_pred == Y_test)
                else:
                    test_acc = 0

            file_name = f'{self.graph_data.name}_Results_run_id_{self.run_num}_validation_step_{self.validation_num}.csv'

            # header use semicolon as delimiter
            header = ("Dataset;RunNumber;ValidationNumber;Algorithm;TrainingSize;ValidationSize;TestSize"
                      ";HyperparameterSVC;HyperparameterAlgo;ValidationAccuracy;TestAccuracy\n")

            # Save file for results and add header if the file is new
            with open(self.out_path.joinpath(file_name), "a") as file_obj:
                if Path.stat(self.out_path.joinpath(file_name)).st_size == 0:
                    file_obj.write(header)

            # Save results to file
            with open(self.out_path.joinpath(file_name), "a") as file_obj:
                file_obj.write(f"{self.graph_data.name};{self.run_num};{self.validation_num};NoGKernel;{len(self.training_data)};{len(self.validate_data)};{len(self.test_data)};{c_param};{0};{val_acc};{test_acc}\n")
