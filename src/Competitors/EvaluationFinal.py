import os

import pandas as pd
from matplotlib import pyplot as plt
from sympy import pretty_print as pp, latex
from sympy.abc import a, b, n


def evaluateGraphLearningNN(db_name, algorithm):
    evaluation = {}

    # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
    # ge all those files
    files = []
    for file in os.listdir(f"Results/"):
        if file.startswith(f"{db_name}_Results_run_id_") and file.endswith(".csv"):
            files.append(file)

    df_all = None
    for i, file in enumerate(files):
        df = pd.read_csv(f"Results/{file}", delimiter=";")
        # concatenate the dataframes
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)

    # get all rows where the algorithm is NoGKernel
    df_all = df_all[df_all['Algorithm'] == algorithm]
    # if the column HyperparameterAlgo is not present, add it with the value 0
    if 'HyperparameterAlgo' not in df_all.columns:
        df_all['HyperparameterAlgo'] = 0

    # group by the hyperparametersvc and the hyperparameterAlgo
    groups = df_all.groupby(['HyperparameterSVC', 'HyperparameterAlgo'])
    # groups = df_all.groupby('HyperparameterSVC')

    evaluation = []
    for name, group in groups:
        df_validation = group.groupby('ValidationNumber').mean(numeric_only=True)
        # get the average and deviation over all runs
        avg = df_validation.mean(numeric_only=True)
        std = df_validation.std()
        evaluation.append(
            {'HyperparameterSVC': avg['HyperparameterSVC'], 'HyperparameterAlgo': avg['HyperparameterAlgo'],
             'ValidationAccuracy': round(100 * avg['ValidationAccuracy'], 2),
             'TestAccuracy': round(100 * avg['TestAccuracy'], 2),
             'TestAccuracyStd': round(100 * std['TestAccuracy'], 2)})
        # print the avg and std together with the hyperparameter and the algorithm used
        print(f"Hyperparameter: {name} Average Test Accuracy: {avg['TestAccuracy']} +/- {std['TestAccuracy']}")

    # get the three best hyperparameters according to the average validation accuracy
    evaluation = sorted(evaluation, key=lambda x: x['ValidationAccuracy'], reverse=True)
    best_hyperparameters = evaluation[:3]
    # print the three best hyperparameters
    print(f"{db_name} Best hyperparameters:")
    for hyperparameter in best_hyperparameters:
        print(
            f"Hyperparameter: SVC:{hyperparameter['HyperparameterSVC']} Algo:{hyperparameter['HyperparameterAlgo']} Validation Accuracy: {hyperparameter['ValidationAccuracy']}Test Accuracy: {hyperparameter['TestAccuracy']} +/- {hyperparameter['TestAccuracyStd']}")


def main():
    evaluateGraphLearningNN(db_name='EvenOddRingsCount16', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='EvenOddRingsCount16', algorithm="NoGKernel")

    evaluateGraphLearningNN(db_name='Snowflakes', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='Snowflakes', algorithm="NoGKernel")

    evaluateGraphLearningNN(db_name='EvenOddRings2_16', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='EvenOddRings2_16', algorithm="NoGKernel")

    evaluateGraphLearningNN(db_name='CSL_original', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='CSL_original', algorithm="NoGKernel")

    evaluateGraphLearningNN(db_name='LongRings100', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='LongRings100', algorithm="NoGKernel")

    evaluateGraphLearningNN(db_name='NCI1', algorithm="NoGKernel")
    evaluateGraphLearningNN(db_name='NCI109', algorithm="NoGKernel")
    evaluateGraphLearningNN(db_name='Mutagenicity', algorithm="NoGKernel")
    evaluateGraphLearningNN(db_name='DHFR', algorithm="NoGKernel")
    evaluateGraphLearningNN(db_name='SYNTHETICnew', algorithm="NoGKernel")
    evaluateGraphLearningNN(db_name='CSL_original', algorithm="NoGKernel")
    evaluateGraphLearningNN(db_name='IMDB-BINARY', algorithm="NoGKernel")
    evaluateGraphLearningNN(db_name='IMDB-MULTI', algorithm="NoGKernel")

    evaluateGraphLearningNN(db_name='DHFR', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='SYNTHETICnew', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='CSL_original', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='NCI1', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='NCI109', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='Mutagenicity', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='IMDB-BINARY', algorithm="WLKernel")
    evaluateGraphLearningNN(db_name='IMDB-MULTI', algorithm="WLKernel")




if __name__ == "__main__":
    main()
