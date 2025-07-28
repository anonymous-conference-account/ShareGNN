import torch

from src.Preprocessing.GraphData.GraphData import NodeLabels


def load_labels(path='') -> NodeLabels:
    """
    Load the labels from a file.
    :param path: Path to the file
    :return: NodeLabels object
    """
    dataset_name, label_name, node_labels = torch.load(path, weights_only=True)
    return NodeLabels(dataset_name, label_name, node_labels)