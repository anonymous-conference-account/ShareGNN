import os

import numpy as np
import torch
import torch_geometric.datasets
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.nn.models import GCN, GIN

from src.Competitors.GNN.Models.GraphSAGE import GraphSAGE
from GraphData.DataSplits.load_splits import Load_Splits
from src.Preprocessing.GraphData.GraphData import get_graph_data, BenchmarkDatasets


class GNNModule(torch.nn.Module):
    def __init__(self, run_configuration, in_channels, out_channels):
        super().__init__()
        self.conv1 = run_configuration["conv_model"](in_channels=in_channels,
                                                     out_channels=run_configuration["hidden_channels"])
        # linear output layer
        self.linear = torch.nn.Linear(run_configuration["hidden_channels"], out_channels)

        self.layers = []
        self.layers.append(self.conv1)
        for i in range(run_configuration["layers"] - 1):
            self.layers.append(run_configuration["conv_model"](in_channels=run_configuration["hidden_channels"],
                                                               out_channels=run_configuration["hidden_channels"]))
        self.layers.append(self.linear)
        self.aggregation = run_configuration["neighborhood_aggregation"]

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = layer(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            else:
                x = self.aggregation(x, batch.batch)
                x = layer(x)

        return F.log_softmax(x, dim=1)


class GNN(torch.nn.Module):
    def __init__(self, db_name, run_num, validation_num, training_data, validate_data, test_data, seed,
                 run_configuration, data_path=None):
        super().__init__()
        self.db_name = db_name
        self.data_path = data_path
        self.run_num = run_num
        self.validation_num = validation_num
        self.training_data = training_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.seed = seed
        self.dataset = self.load_dataset()
        self.run_configuration = run_configuration

    def load_dataset(self):
        # load the data using TUDataset from PyTorch Geometric
        # create Datasets folder if it does not exist
        dataset = None
        if not os.path.exists('Datasets'):
            os.makedirs('Datasets')
        try:
            dataset = torch_geometric.datasets.TUDataset(root='Datasets', name=self.db_name)
        except:
            graph_data = get_graph_data(self.db_name, data_path=self.data_path)
            dataset = BenchmarkDatasets(root=f'../../GraphBenchmarks/BenchmarkGraphs', name=self.db_name, graph_data=graph_data)
        return dataset

    def ModelSelection(self, model):
        # get training, validation and test data
        training_data = self.dataset[self.training_data]
        validate_data = self.dataset[self.validate_data]
        test_data = self.dataset[self.test_data]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.run_configuration["learning_rate"])

        model.train()
        for epoch in range(self.run_configuration["epochs"]):
            loader = torch_geometric.loader.DataLoader(training_data, batch_size=self.run_configuration["batch_size"], shuffle=True)
            epoch_loss = 0
            epoch_accuracy = 0
            for batch in loader:
                optimizer.zero_grad()
                out = model(batch)
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                # get epoch accuracy
                pred = out.argmax(dim=1)
                acc = accuracy_score(batch.y.cpu().numpy(), pred.cpu().numpy())
                optimizer.step()
                epoch_loss += loss.item() * len(batch) / len(training_data)
                epoch_accuracy += acc * len(batch) / len(training_data)

            # get the validation accuracy
            model.eval()
            loader = torch_geometric.loader.DataLoader(validate_data, batch_size=self.validate_data.size,
                                                       shuffle=False)
            validation_accuracy = 0
            validation_loss = 0
            for batch in loader:
                out = model(batch)
                loss = F.nll_loss(out, batch.y)
                pred = out.argmax(dim=1)
                acc = accuracy_score(batch.y, pred)
                validation_accuracy += acc * len(batch) / len(validate_data)
                validation_loss += loss.item() * len(batch) / len(validate_data)
            # round the accuracy to 4 decimal places
            validation_accuracy = round(validation_accuracy, 4)
            validation_loss = round(validation_loss, 4)
            # print epoch, batch loss and validation accuracy
            print(
                f'Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Validation Accuracy: {validation_accuracy}, Validation Loss: {validation_loss}')
            # write to file
            file_name = f'{self.db_name}_Results_run_id_{self.run_num}_validation_step_{self.validation_num}.csv'
            with open(f'Results/{file_name}', "a") as file_obj:
                file_obj.write(
                    f"{self.db_name};{self.run_num};{self.validation_num};GNN;{len(self.training_data)};{len(self.validate_data)};{len(self.test_data)};0;0;{validation_accuracy};0\n")


    def BasicModelSelection(self, model):
        # get training, validation and test data
        training_data = self.dataset[self.training_data]
        validate_data = self.dataset[self.validate_data]
        test_data = self.dataset[self.test_data]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.run_configuration["learning_rate"])

        model.train()
        for epoch in range(self.run_configuration["epochs"]):
            loader = torch_geometric.loader.DataLoader(training_data, batch_size=self.run_configuration["batch_size"], shuffle=True)
            epoch_loss = 0
            epoch_accuracy = 0
            for batch in loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                # get epoch accuracy
                pred = out.argmax(dim=1)
                acc = accuracy_score(batch.y.cpu().numpy(), pred.cpu().numpy())
                optimizer.step()
                epoch_loss += loss.item() * len(batch) / len(training_data)
                epoch_accuracy += acc * len(batch) / len(training_data)

            # get the validation accuracy
            model.eval()
            loader = torch_geometric.loader.DataLoader(validate_data, batch_size=self.validate_data.size,
                                                       shuffle=False)
            validation_accuracy = 0
            validation_loss = 0
            for batch in loader:
                out = model(batch.x, batch.edge_index)
                loss = F.nll_loss(out, batch.y)
                pred = out.argmax(dim=1)
                acc = accuracy_score(batch.y, pred)
                validation_accuracy += acc * len(batch) / len(validate_data)
                validation_loss += loss.item() * len(batch) / len(validate_data)
            # round the accuracy to 4 decimal places
            validation_accuracy = round(validation_accuracy, 4)
            validation_loss = round(validation_loss, 4)
            # print epoch, batch loss and validation accuracy
            print(
                f'Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Validation Accuracy: {validation_accuracy}, Validation Loss: {validation_loss}')
            # write to file
            file_name = f'{self.db_name}_Results_run_id_{self.run_num}_validation_step_{self.validation_num}.csv'
            with open(f'Results/{file_name}', "a") as file_obj:
                file_obj.write(
                    f"{self.db_name};{self.run_num};{self.validation_num};GNN;{len(self.training_data)};{len(self.validate_data)};{len(self.test_data)};0;0;{validation_accuracy};0\n")


def main(db_name, data_path=None):
    validation_size = 10
    run_number = 0
    run_seed = 368787 + run_number

    run_configuration = {"conv_model": GCNConv, "layers": 8, "batch_size": 32, "epochs": 1000, "learning_rate": 0.001,
                         "hidden_channels": 64, "neighborhood_aggregation": global_max_pool}

    if db_name == "CSL_original":
        validation_size = 5
    for validation_id in range(validation_size):

        # create results file
        file_name = f'{db_name}_Results_run_id_{run_number}_validation_step_{validation_id}.csv'

        # header use semicolon as delimiter
        header = ("Dataset;RunNumber;ValidationNumber;Algorithm;TrainingSize;ValidationSize;TestSize"
                  ";HyperparameterSVC;HyperparameterAlgo;ValidationAccuracy;TestAccuracy\n")
        # Save file for results and add header if the file is new
        with open(f'Results/{file_name}', "a") as file_obj:
            if os.stat(f'Results/{file_name}').st_size == 0:
                file_obj.write(header)

        data = Load_Splits("../../BenchmarkGraphs/Splits", db_name)
        test_data = np.asarray(data[0][validation_id], dtype=int)
        training_data = np.asarray(data[1][validation_id], dtype=int)
        validate_data = np.asarray(data[2][validation_id], dtype=int)

        gnn = GNN(db_name, run_num=run_number, validation_num=validation_id, training_data=training_data,
                  validate_data=validate_data, test_data=test_data, seed=run_seed, data_path=data_path,
                  run_configuration=run_configuration)
        model_GIN = GIN(in_channels=gnn.dataset.num_node_features, hidden_channels=16,
                        out_channels=gnn.dataset.num_classes, num_layers=2)
        model_GCN = GCN(in_channels=gnn.dataset.num_node_features, hidden_channels=16,
                        out_channels=gnn.dataset.num_classes, num_layers=2)
        model = GNNModule(run_configuration, in_channels=gnn.dataset.num_node_features,out_channels=gnn.dataset.num_classes)
        model_gs = GraphSAGE(in_channels=gnn.dataset.num_node_features,out_channels=gnn.dataset.num_classes, run_config=run_configuration)
        gnn.ModelSelection(model_gs)
        #gnn.BasicModelSelection(model)
        model_GraphSAGE = GraphSAGE(in_channels=gnn.dataset.num_node_features, hidden_channels=16,
                                    out_channels=gnn.dataset.num_classes, num_layers=2, jk="max")
        model = GNNModule(run_configuration=run_configuration, in_channels=gnn.dataset.num_node_features,out_channels=gnn.dataset.num_classes)
        gnn.ModelSelection(model)
        #gnn.BasicModelSelection(model)


if __name__ == "__main__":
    main("DHFR", data_path="../../../Testing/BenchmarkGraphs/")
    main("EvenOddRingsCount16", data_path="../../../Testing/BenchmarkGraphs/")
