from pathlib import Path

import click

from src.Experiment.ExperimentMain import ExperimentMain


def get_existing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("paper_experiments/classification/Data").mkdir(exist_ok=True)
    Path("paper_experiments/classification/Data/Splits").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "CSL"]:
        source_path = Path("Data/Splits").joinpath(f"{split}_splits.json")
        target_path = Path("paper_experiments/classification/Data/Splits").joinpath(f"{split}_splits.json")
        target_path.write_text(source_path.read_text())

    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("paper_experiments/classification/Data/SplitsSimple").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]:
        source_path = Path("Data/SplitsSimple").joinpath(f"{split}_splits.json")
        target_path = Path("paper_experiments/classification/Data/SplitsSimple").joinpath(f"{split}_splits.json")
        target_path.write_text(source_path.read_text())

def main_ablation_distance(num_threads=-1):
    get_existing_splits()
    ablation_experiment = ExperimentMain(Path(f'paper_experiments/classification/configs/ablation/distances/main_config_ablation_distances.yml'))
    ablation_experiment.ExperimentPreprocessing(num_threads=num_threads)
    ablation_experiment.GridSearch(num_threads=num_threads)
    ablation_experiment.EvaluateResults()
    ablation_experiment.RunBestModel(num_threads=num_threads)
    ablation_experiment.EvaluateResults(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_ablation_distance(num_threads)


if __name__ == '__main__':
    main()