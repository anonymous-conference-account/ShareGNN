# download ZINC_original data a subset of ZINC_original dataset
from pathlib import Path

from torch_geometric.datasets import ZINC

from src.Preprocessing.GraphData.GraphData import zinc_to_graph_data
from src.utils.utils import save_graphs


def main(pytorch_geometric=None):
    zinc_train = ZINC(root="Data/BenchmarkGraphs/ZINC_original/", subset=True, split='train')
    zinc_val = ZINC(root="Data/BenchmarkGraphs/ZINC_original/", subset=True, split='val')
    zinc_test = ZINC(root="Data/BenchmarkGraphs/ZINC_original/", subset=True, split='test')
    # zinc to networkx
    networkx_graphs = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC_original")
    save_graphs(db_name="ZINC", graphs=networkx_graphs.graphs, path=Path("Data/BenchmarkGraphs/"), labels=networkx_graphs.graph_labels)
    pass
    
    
if __name__ == "__main__":
    main()