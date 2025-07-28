import json
import pickle


def load_distances(db_name, path=None):
    """
    Load the distances from a file.

    :param file: str
    :return: list of dictionaries
    """
    if path is None:
        distances = pickle.load(open(f"{db_name}_distances.pkl", 'rb'))
        return distances
    else:
        distances = pickle.load(open(f"{path}", 'rb'))
        return distances



if __name__ == '__main__':
    load_distances("NCI1")