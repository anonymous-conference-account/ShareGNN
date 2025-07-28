import os

from src.Architectures.ShareGNN.ShareGNNLayers import get_label_string
from src.Preprocessing.GraphData.GraphData import ShareGNNDataset
from src.utils.GraphLabels import Properties
from src.Architectures.ShareGNN.Parameters import Parameters
from src.Preprocessing.load_labels import load_labels
from src.utils import ReadWriteGraphs as gdtgl


def load_preprocessed_data_and_parameters(run_id, validation_id, config_id, validation_folds, graph_data:ShareGNNDataset, run_config, para: Parameters):
    experiment_configuration = run_config.config
    # path do db and db
    draw = False
    print_results = False
    save_weights = False
    save_prediction_values = False
    plot_graphs = False
    print_layer_init = False
    # if debug mode is on, turn on all print and draw options
    if experiment_configuration['mode'] == "debug":
        draw = experiment_configuration['additional_options']['draw']
        print_results = experiment_configuration['additional_options']['print_results']
        save_prediction_values = experiment_configuration['additional_options']['save_prediction_values']
        save_weights = experiment_configuration['additional_options']['save_weights']
        plot_graphs = experiment_configuration['additional_options']['plot_graphs']

    unique_label_dicts = []
    label_layer_ids = []
    unique_properties = []
    properties_layer_ids = []
    for i, l in enumerate(run_config.layers):
        # add the labels to the graph data
        new_unique = l.get_unique_layer_dicts()
        for x in new_unique:
            if x not in unique_label_dicts:
                unique_label_dicts.append(x)
        property_dicts = l.get_unique_property_dicts()
        if property_dicts:
            for x in property_dicts:
                property_dict_name = x.get_property_string()
                if property_dict_name not in unique_properties:
                    unique_properties.append(property_dict_name)

    for label_dict in unique_label_dicts:
        label_path = experiment_configuration['paths']['labels'].joinpath(f'{graph_data.name}').joinpath(f"{graph_data.name}_labels_{get_label_string(label_dict)}.pt")
        if os.path.exists(label_path):
            g_labels = load_labels(path=label_path)
            graph_data.node_labels[get_label_string(label_dict)] = g_labels
        else:
            # raise an error if the file does not exist and add the absolute path to the error message
            raise FileNotFoundError(f"File {label_path} does not exist")

    for prop_name in unique_properties:
        valid_values = {}
        for i, l in enumerate(run_config.layers):
            for j, c in enumerate(l.layer_heads):
                if c.property_dict.property_dict is not None:
                    if c.property_dict.get_property_string() == prop_name:
                        valid_values[(i,j)] = c.property_dict.get_values()

        graph_data.properties[prop_name] = Properties(path=experiment_configuration['paths']['properties'], db_name=graph_data.name,
                                                      property_name=prop_name,
                                                      valid_values=valid_values)

    """
        BenchmarkGraphs parameters
    """
    para.set_data_param(db=graph_data.name,
                        max_coding=1,
                        layers=run_config.layers, node_features=1,
                        run_config=run_config)

    """
        Network parameters
    """
    para.set_evaluation_param(run_id=run_id, n_val_runs=validation_folds,
                              validation_id=validation_id,
                              config_id=config_id,
                              n_epochs=run_config.epochs,
                              learning_rate=run_config.lr,
                              dropout=run_config.dropout,
                              balance_data=run_config.config.get('balance_data', False),
                              convolution_grad=True,
                              resize_graph=True)

    """
    Print, save and draw parameters
    """
    para.set_print_param(no_print=False, print_results=print_results, net_print_weights=True, print_number=1,
                         draw=draw, save_weights=save_weights,
                         save_prediction_values=save_prediction_values, plot_graphs=plot_graphs,
                         print_layer_init=print_layer_init)

    """
        Get the first index in the results directory that is not used
    """
    para.set_file_index(size=6)

    if para.plot_graphs:
        # if not exists create the directory
        if not os.path.exists(experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots")):
            os.makedirs(experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots"))
        for i in range(0, len(graph_data.graphs)):
            gdtgl.draw_graph(graph_data.graphs[i], graph_data.graph_labels[i],
                             experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots/graph_{str(i).zfill(5)}.png"))
