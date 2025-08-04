import joblib
import numpy as np

from src.Preprocessing.GraphData.GraphData import get_graph_data
from Kernels.NoGKernel import NoGKernel
from Kernels.GraphKernels import WLKernel

def validation(db_name, validation_id, graph_data):
    # three runs
    for i in range(3):
        data = Load_Splits("../BenchmarkGraphs/Splits", db_name)
        test_data = np.asarray(data[0][validation_id], dtype=int)
        training_data = np.asarray(data[1][validation_id], dtype=int)
        validate_data = np.asarray(data[2][validation_id], dtype=int)

        wlKernel = WLKernel(graph_data, run_num=i, validation_num=validation_id,
                            training_data=training_data, validate_data=validate_data, test_data=test_data,
                            seed=i)
        wlKernel.Run()

        noG = NoGKernel(graph_data, run_num=i, validation_num=validation_id, training_data=training_data,
                        validate_data=validate_data, test_data=test_data, seed=i)
        noG.Run()
def main(db_name, data_path="../../BenchmarkGraphs/DS_all/"):
    # load the graph data
    graph_data = get_graph_data(db_name, data_path=data_path)

    validation_size = 10
    if db_name == "CSL_original":
        validation_size = 5
    # run the validation for all validation sets in parallel
    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(validation)(db_name, validation_id, graph_data) for validation_id in range(validation_size))




if __name__ == "__main__":
    # run parallel for all datasets
    # joblib.Parallel(n_jobs=-1)(
    #     joblib.delayed(main)(db_name) for db_name in ['CSL_original', 'DHFR', 'SYNTHETICnew', 'NCI1', 'NCI109', 'Mutagenicity'])
    #main("IMDB-MULTI")
    #main("SnowflakesCount", data_path="../GraphBenchmarks/BenchmarkGraphs/")
    main("Snowflakes", data_path="../../Testing/BenchmarkGraphs/")
