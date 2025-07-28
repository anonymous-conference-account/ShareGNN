import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.Preprocessing.GraphData.GraphData import get_graph_data, GraphData
from src.utils.utils import get_train_validation_test_list, get_data_indices


def create_transfer_splits(db_name, path:Path, output_path:Path, data_format=None, split_type='random'):
    '''
    Create splits for transfer learning
    :param db_name: name of the dataset
    :param path: path to the dataset
    :param output_path: path to save the splits
    :param data_format: format of the dataset
    :param split_type: type of the split, transfer takes all but the last graph dataset as training and the last as test, mixed takes training and testing equally from all graph datasets, random takes random graphs from all datasets
    '''

    # string to int
    int_type = 0
    for i in range(len(split_type)):
        int_type += ord(split_type[i])
    seed = 687384987 + int_type
    np.random.seed(seed)
    if split_type == "random":
        create_splits(db_name, path, output_path, data_format)
        return
    k = 10
    graph_data = get_graph_data(db_name, path, graph_format=data_format, only_graphs=True)
    # number of graphs per graph_dataset
    graph_number_map = defaultdict(list)
    graph_datasets = []
    for i, graph in enumerate(graph_data.graphs):
        graph_number_map[graph.name].append(i)
        if graph.name not in graph_datasets:
            graph_datasets.append(graph.name)

    if split_type == "transfer":
        test_indices = graph_number_map[graph_datasets[-1]]
        # duplicate test_data k times
        test_data = [test_indices for i in range(k)]
        # shuffle the training data and create k folds
        training_indices = []
        for dataset in graph_datasets[:-1]:
            training_indices += graph_number_map[dataset]
        training_indices = np.random.permutation(training_indices)
        # split the training data into k folds
        data_splits = np.array_split(training_indices, k)

        splits = []
        for i in range(k):
            training_data = []
            validate_data = []
            # append all the training data except the validation data
            training_data += [data for j, data in enumerate(data_splits) if j != i]
            # flatten the list and sort it
            training_data = np.sort(np.concatenate(training_data)).tolist()
            # append the validation data
            # sort the validation data
            validate_data = np.sort(data_splits[i]).tolist()
            splits.append(
                {"test": test_data[i], "model_selection": [{"train": training_data, "validation": validate_data}]})

        # save splits to json as one line use json.dumps
        with open(output_path.joinpath(f"{db_name}_splits_transfer.json"), "w") as f:
            f.write(json.dumps(splits))
    elif split_type == "mixed":
        # split the graph_datasets equally into k parts
        parts = []
        for i in range(len(graph_datasets)):
            shuffled_indices = np.random.permutation(graph_number_map[graph_datasets[i]])
            splits = np.array_split(shuffled_indices, k)
            # sort the splits
            for i, split in enumerate(splits):
                splits[i] = np.sort(split)
            parts.append(splits)
        run_test_indices = []
        for i in range(k):
            run_test_indices.append([])
            for j in parts:
                run_test_indices[i] += j[i].tolist()

        splits = []
        for validation_id in range(0, k):
            seed = 687384987 + validation_id + k

            """
            Create the data
            """
            training_data, validate_data, test_data = get_train_validation_test_list(test_indices=run_test_indices,
                                                                                         validation_step=validation_id,
                                                                                         seed=seed,
                                                                                         balanced=False,
                                                                                         graph_labels=graph_data.graph_labels,
                                                                                         val_size=1.0 / k)

            # Dict use double quotes
            training_data = [int(x) for x in training_data]
            validate_data = [int(x) for x in validate_data]
            test_data = [int(x) for x in test_data]
            # write data to txt file
            # with open(f"{output_path}{db_name}_train.txt", "a") as f:
            #    f.write(" ".join([str(x) for x in training_data]))
            #    f.write("\n")
            # with open(f"{output_path}{db_name}_validation.txt", "a") as f:
            #    f.write(" ".join([str(x) for x in validate_data]))
            #    f.write("\n")
            # with open(f"{output_path}{db_name}_test.txt", "a") as f:
            #    f.write(" ".join([str(x) for x in test_data]))
            #    f.write("\n")

            splits.append(
                {"test": test_data, "model_selection": [{"train": training_data, "validation": validate_data}]})

        # save splits to json as one line use json.dumps
        with open(output_path.joinpath(f"{db_name}_splits_mixed.json"), "w") as f:
            f.write(json.dumps(splits))

def splits_from_index_lists(training_indices, validation_indices, test_indices, db_name, output_path):
    splits = []
    folds = len(training_indices)
    if len(validation_indices) != folds or len(test_indices) != folds:
        raise ValueError("Length of training, validation and test indices should be the same")
    for i in range(0, folds):
        splits.append({"test": test_indices[i], "model_selection": [{"train": training_indices[i], "validation": validation_indices[i]}]})
    # save splits to json as one line use json.dumps
    # check if the output path exists
    if not output_path.joinpath(f"{db_name}_splits.json").exists():
        print(f"Creating new split file at {output_path.joinpath(f'{db_name}_splits.json')}")
        with open(output_path.joinpath(f"{db_name}_splits.json"), "w") as f:
            f.write(json.dumps(splits))
    else:
        print(f"File {output_path.joinpath(f'{db_name}_splits.json')} already exists. Skipping new split creation.")


def create_splits(db_name: str, data_path: Path = Path("../GraphData/DS_all/"), output_path: Path=Path("Data/Splits/"), folds=10, graph_data=None, graph_format=None, seed=2045287)->Path:
    splits = []
    if graph_data is None:
        graph_data = get_graph_data(db_name=db_name, data_path=data_path, graph_format=graph_format, only_graphs=True)

    run_test_indices = get_data_indices(len(graph_data), seed=seed, kFold=folds)
    for validation_id in range(0, folds):
        validation_seed = seed + validation_id

        """
        Create the data
        """
        graph_labels = None
        if type(graph_data) == GraphData:
            graph_labels = graph_data.graph_labels
        else:
            graph_labels = graph_data.y

        training_data, validate_data, test_data = get_train_validation_test_list(test_indices=run_test_indices,
                                                                                         validation_step=validation_id,
                                                                                         seed=validation_seed,
                                                                                         balanced=False,
                                                                                         graph_labels=graph_labels,
                                                                                         val_size=1.0 / folds)
        # Dict use double quotes
        training_data = [int(x) for x in training_data]
        validate_data = [int(x) for x in validate_data]
        test_data = [int(x) for x in test_data]
        # write data to txt file
        #with open(f"{output_path}{db_name}_train.txt", "a") as f:
        #    f.write(" ".join([str(x) for x in training_data]))
        #    f.write("\n")
        #with open(f"{output_path}{db_name}_validation.txt", "a") as f:
        #    f.write(" ".join([str(x) for x in validate_data]))
        #    f.write("\n")
        #with open(f"{output_path}{db_name}_test.txt", "a") as f:
        #    f.write(" ".join([str(x) for x in test_data]))
        #    f.write("\n")

        splits.append({"test": test_data, "model_selection": [{"train": training_data, "validation": validate_data}]})

    # save splits to json as one line use json.dumps
    # check if the output path exists
    if not output_path.joinpath(f"{db_name}_splits.json").exists():
        print(f"Creating new split file at {output_path.joinpath(f'{db_name}_splits.json')}")
        with open(output_path.joinpath(f"{db_name}_splits.json"), "w") as f:
            f.write(json.dumps(splits))
        return output_path.joinpath(f"{db_name}_splits.json")
    else:
        print(f"File {output_path.joinpath(f'{db_name}_splits.json')} already exists. Skipping new split creation.")
    return None


def splits_from_train_test_files(path:Path, db_name:str, output_path:Path = None):
    '''
    Convert the train, test files given by https://github.com/weihua916/powerful-gnns into our json split file format
    :param path: path to the train, test files
    '''
    # get all the files in path that are of type train_idx-*.txt or test_idx-*.txt where * is an arbitrary number
    train_files = list(path.glob("train_idx-*.txt"))
    test_files = list(path.glob("test_idx-*.txt"))
    # sort the files
    train_files.sort()
    test_files.sort()
    if len(train_files) != len(test_files):
        raise ValueError("Number of train and test files should be the same")
    splits = []
    for i in range(len(train_files)):
        training_data = []
        validate_data = []
        test_data = []
        # read the training data (each line is an idx)
        with open(train_files[i], "r") as f:
            for line in f:
                training_data.append(int(line.strip()))
        # read the test data (each line is an idx)
        with open(test_files[i], "r") as f:
            for line in f:
                validate_data.append(int(line.strip()))
        # sort the training data
        training_data = np.sort(training_data).tolist()
        validate_data = np.sort(validate_data).tolist()


        splits.append({"test": test_data, "model_selection": [{"train": training_data, "validation": validate_data}]})
    # save splits to json as one line use json.dumps
    # check if the output path exists
    if not output_path.joinpath(f"{db_name}_splits.json").exists():
        print(f"Creating new split file at {output_path.joinpath(f'{db_name}_splits.json')}")
        with open(output_path.joinpath(f"{db_name}_splits.json"), "w") as f:
            f.write(json.dumps(splits))
    else:
        print(f"File {output_path.joinpath(f'{db_name}_splits.json')} already exists. Skipping new split creation.")

def splits_from_train_test_files_repair_NCI109(path:Path, db_name:str, output_path:Path = None):
    '''
    Convert the train, test files given by https://github.com/weihua916/powerful-gnns into our json split file format
    :param path: path to the train, test files
    '''
    # get all the files in path that are of type train_idx-*.txt or test_idx-*.txt where * is an arbitrary number
    train_files = list(path.glob("train_idx-*.txt"))
    test_files = list(path.glob("test_idx-*.txt"))
    # sort the files
    train_files.sort()
    test_files.sort()
    splits = []
    for i in range(len(test_files)):
        training_data = []
        validate_data = []
        test_data = []

        # read the test data (each line is an idx)
        with open(test_files[i], "r") as f:
            for line in f:
                validate_data.append(int(line.strip()))
        # sort the validation data
        validate_data = np.sort(validate_data).tolist()

        if i >= 8:
            max_id = 4127
            # get all ids not in the validation data
            training_data = [x for x in range(max_id) if x not in validate_data]
        else:
            # read the training data (each line is an idx)
            with open(train_files[i], "r") as f:
                for line in f:
                    training_data.append(int(line.strip()))
        # sort the training data
        training_data = np.sort(training_data).tolist()




        splits.append({"test": test_data, "model_selection": [{"train": training_data, "validation": validate_data}]})
    # save splits to json as one line use json.dumps
    # check if the output path exists
    if not output_path.joinpath(f"{db_name}_splits.json").exists():
        print(f"Creating new split file at {output_path.joinpath(f'{db_name}_splits.json')}")
        with open(output_path.joinpath(f"{db_name}_splits.json"), "w") as f:
            f.write(json.dumps(splits))
    else:
        print(f"File {output_path.joinpath(f'{db_name}_splits.json')} already exists. Skipping new split creation.")


if __name__ == "__main__":

    for db in ['NCI1', 'NCI109', 'IMDBBINARY', 'IMDBMULTI']:
        source_path = Path(f"Data/SOTA_Data/{db}/raw/10fold_idx")
        if db == 'IMDBBINARY':
            db = 'IMDB-BINARY'
        elif db == 'IMDBMULTI':
            db = 'IMDB-MULTI'

        if db == 'NCI109':
            splits_from_train_test_files_repair_NCI109(source_path, db, Path("Data/SplitsSimple/"))
        else:
            splits_from_train_test_files(source_path, db, Path("Data/SplitsSimple/"))

    #zinc_splits()
    #create_splits("DHFR")
    #create_splits("Mutagenicity")
    #create_splits("NCI109")
    #create_splits("SYNTHETICnew")
    # for db in ["DHFR", "Mutagenicity", "NCI109", "SYNTHETICnew", "MUTAG"]:
    #     create_splits(db)
    #for db in ['PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR']:
    #    create_splits(db)
