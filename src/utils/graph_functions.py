import networkx as nx
from typing import List, Dict, Tuple, Any
import numpy as np
# import the GED using the munkres algorithm
import gmatch4py as gm

def pairwise_ged_paths(graphs: List[nx.Graph], node_match=None, edge_match=None) -> Dict[Tuple[int, int], Tuple[float, List[Any]]]:
    """
    Compute pairwise Graph Edit Distance (GED) paths for a list of NetworkX graphs.
    
    Args:
        graphs (List[nx.Graph]): List of NetworkX graphs to compute pairwise GED paths for.
        node_match (callable, optional): A function that compares node attributes. 
            The function should accept two node attribute dictionaries as inputs and return True if the attributes match, False otherwise.
            Default is None, which means node attributes are not considered.
        edge_match (callable, optional): A function that compares edge attributes.
            The function should accept two edge attribute dictionaries as inputs and return True if the attributes match, False otherwise.
            Default is None, which means edge attributes are not considered.
    
    Returns:
        Dict[Tuple[int, int], Tuple[float, List[Any]]]: A dictionary mapping pairs of graph indices to their GED and edit path.
            The key is a tuple of (i, j) where i and j are indices of graphs in the input list.
            The value is a tuple of (distance, path) where distance is the GED and path is the edit path.
    """
    results = {}
    n = len(graphs)
    ged = gm.GraphEditDistance(1,1,1,1)
    for i in range(n):
        for j in range(i + 1, n):
            # Compute the Graph Edit Distance between graphs i and j
            results[(i,j)] = ged.edit_path(graphs[i], graphs[j])
    results = ged.compare(graphs, None)
    
    return results

def visualize_ged_matrix(ged_results: Dict[Tuple[int, int], Tuple[float, List[Any]]], n_graphs: int) -> np.ndarray:
    """
    Create a matrix visualization of the Graph Edit Distances.
    
    Args:
        ged_results (Dict[Tuple[int, int], Tuple[float, List[Any]]]): The results from pairwise_ged_paths.
        n_graphs (int): The number of graphs.
    
    Returns:
        np.ndarray: A matrix where element (i, j) is the GED between graphs i and j.
    """
    matrix = np.zeros((n_graphs, n_graphs))
    
    for (i, j), (distance, _) in ged_results.items():
        matrix[i, j] = distance
    
    return matrix