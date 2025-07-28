import json
import pickle


def Load_Splits(path, db_name, appendix=None):
    """
    Load the splits for a given database.
    :param path: Base path where the splits are stored. _splits + db_name + .json is added to the path.
    :param db_name: database name, e.g., NCI1, IMDB-BINARY, IMDB-MULTI, CSL.
    :param appendix: Optional appendix to the file name, e.g., 'transfer', the function loads adds "_transfer" to the file name (before the .json extension).
    :return: Cross-validation splits for the given database consisting of three lists: The first one is the list of test indices, the second one is the list of training indices, and the third one is the list of validation indices.
    Each sublist contains the indices of the graphs in the respective split.
    """
    splits = None
    if appendix is not None:
        appendix = f"{appendix}_"
    else:
        appendix = ""

    with open(f"{path}/{db_name}_{appendix}splits.json", "rb") as f:
        splits = json.load(f)

    test_indices = [x['test'] for x in splits]
    train_indices = [x['model_selection'][0]['train'] for x in splits]
    vali_indices = [x['model_selection'][0]['validation'] for x in splits]

    return test_indices, train_indices, vali_indices