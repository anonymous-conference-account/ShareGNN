#
import os
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from grakel import WeisfeilerLehman, VertexHistogram
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_absolute_error

from src.Preprocessing.GraphData.GraphData import ShareGNNDataset


def nx_to_grakel(nx_graphs: List[nx.Graph]):
    # create input for the kernel from the grakel graphs
    grakel_graphs = []
    for g in nx_graphs:
        edge_set = set()
        edge_dict = {}
        edges = g.edges(data=True)
        for e in edges:
            label = 0
            if 'primary_edge_labels' in e[2]:
                label = e[2]['primary_edge_labels']
                if isinstance(label, list) and len(label) == 1:
                    label = label[0]
                    try:
                        label = int(label)
                    except:
                        label = 0
            else:
                label = 0
            edge_dict[(e[0], e[1])] = label
            edge_dict[(e[1], e[0])] = label
            edge_set.add((e[0], e[1]))
            edge_set.add((e[1], e[0]))
        node_dict = {}
        for n in g.nodes(data=True):
            label = 0
            if 'primary_node_labels' in n[1]:
                label = n[1]['primary_node_labels']
                if isinstance(label, list) and len(label) == 1:
                    label = label[0]
                    try:
                        label = int(label)
                    except:
                        label = 0
            else:
                label = 0
            node_dict[n[0]] = label
        grakel_graphs.append([edge_set, node_dict, edge_dict])
    return grakel_graphs


class WLKernel:
    def __init__(self, out_path:Path, graph_data:ShareGNNDataset, run_num: int, validation_num: int, training_data: List[int],
                 validate_data: List[int], test_data: List[int],
                 seed: int):
        self.out_path = out_path
        self.graph_data = graph_data
        self.graph_data.create_nx_graphs()
        self.training_data = training_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.seed = seed
        self.run_num = run_num
        self.validation_num = validation_num


    def Run(self):

        train_graphs = [self.graph_data.nx_graphs[i] for i in self.training_data]
        val_graphs = [self.graph_data.nx_graphs[i] for i in self.validate_data]
        test_graphs = [self.graph_data.nx_graphs[i] for i in self.test_data]

        y_train = np.asarray([self.graph_data.y[i].item() for i in self.training_data])
        y_val = np.asarray([self.graph_data.y[i].item()for i in self.validate_data])
        y_test = np.asarray([self.graph_data.y[i].item() for i in self.test_data])

        grakel_train = nx_to_grakel(train_graphs)
        grakel_val = nx_to_grakel(val_graphs)
        grakel_test = nx_to_grakel(test_graphs)

        for n_iter in range(1, 15):
            for c_param in [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11]:
                c_param = 2 ** c_param
                gk = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=True)
                K_train = gk.fit_transform(grakel_train)
                K_val = gk.transform(grakel_val)
                if len(grakel_test) > 0:
                    K_test = gk.transform(grakel_test)

                if type(y_train) is not np.ndarray:
                    clf = SVR(kernel='precomputed', C=c_param)
                    clf = MultiOutputRegressor(clf)
                else:
                    clf = SVC(C=c_param, kernel="precomputed", random_state=self.seed)
                # Uses the SVM classifier to perform classification
                clf.fit(K_train, y_train)
                y_val_pred = clf.predict(K_val)
                if len(grakel_test) > 0:
                    y_test_pred = clf.predict(K_test)

                if type(y_train) is not np.ndarray:
                    val_acc = mean_absolute_error(y_val, y_val_pred)
                    if len(grakel_test) > 0:
                        test_acc = mean_absolute_error(y_test, y_test_pred)
                    else:
                        test_acc = 0
                else:
                    # compute the validation accuracy and test accuracy and print it
                    val_acc = accuracy_score(y_val, y_val_pred)
                    if len(grakel_test) > 0:
                        test_acc = accuracy_score(y_test, y_test_pred)
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
                    file_obj.write(
                        f"{self.graph_data.name};{self.run_num};{self.validation_num};WLKernel;{len(self.training_data)};{len(self.validate_data)};{len(self.test_data)};{c_param};{n_iter};{val_acc};{test_acc}\n")
