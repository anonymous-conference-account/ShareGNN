from pathlib import Path
import click

from src.Experiment.ExperimentMain import ExperimentMain


def main_synthetic(num_threads=-1):
    ## Synthetic Data
    experiment_synthetic = ExperimentMain(Path('paper_experiments/classification/configs/main_config_fair_synthetic.yml'))
    experiment_synthetic.ExperimentPreprocessing(num_threads=num_threads)

    ## run synthetic experiment
    experiment_synthetic.GridSearch(num_threads=num_threads)
    experiment_synthetic.EvaluateResults()
    experiment_synthetic.RunBestModel(num_threads=num_threads)
    experiment_synthetic.EvaluateResults(evaluate_best_model=True)

    experiment_synthetic = ExperimentMain(Path('paper_experiments/classification/configs/main_config_fair_synthetic_random_variation.yml'))
    experiment_synthetic.ExperimentPreprocessing(num_threads=num_threads)
    experiment_synthetic.GridSearch(num_threads=num_threads)
    experiment_synthetic.EvaluateResults()
    experiment_synthetic.RunBestModel(num_threads=num_threads)
    experiment_synthetic.EvaluateResults(evaluate_best_model=True)

    experiment_synthetic = ExperimentMain(Path('paper_experiments/classification/configs/main_config_fair_synthetic_only_encoder.yml'))
    experiment_synthetic.ExperimentPreprocessing(num_threads=num_threads)
    experiment_synthetic.GridSearch(num_threads=num_threads)
    experiment_synthetic.EvaluateResults()
    experiment_synthetic.RunBestModel(num_threads=num_threads)
    experiment_synthetic.EvaluateResults(evaluate_best_model=True)

    experiment_synthetic = ExperimentMain(Path('paper_experiments/classification/configs/main_config_fair_synthetic_only_decoder.yml'))
    experiment_synthetic.ExperimentPreprocessing(num_threads=num_threads)
    experiment_synthetic.GridSearch(num_threads=num_threads)
    experiment_synthetic.EvaluateResults()
    experiment_synthetic.RunBestModel(num_threads=num_threads)
    experiment_synthetic.EvaluateResults(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_synthetic(num_threads)




if __name__ == '__main__':
    main()