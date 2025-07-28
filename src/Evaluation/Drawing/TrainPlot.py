import os

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import pandas.plotting as pdplot


def epoch_accuracy(db_name, y_val):
    if y_val == 'Train':
        y_val = 'EpochAccuracy'
        size = 'TrainingSize'
    elif y_val == 'Validation':
        y_val = 'ValidationAccuracy'
        size = 'ValidationSize'
    elif y_val == 'Test':
        y_val = 'TestAccuracy'
        size = 'TestSize'

    print(f"Model Selection Evaluation for {db_name}")
    evaluation = {}

    # get all ids
    ids = []
    for file in os.listdir(f"../TEMP/Train/{db_name}/Results"):
        if file.endswith(".txt"):
            id = int(file.split('_')[-2])
            ids.append(id)
    # sort the ids
    ids.sort()

    df_all = None

    for id in ids:
        id_str = str(id).zfill(6)
        # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
        # ge all those files
        files = []
        network_files = []
        for file in os.listdir(f"../TEMP/Train/{db_name}/Results"):
            if file.startswith(f"{db_name}_Configuration_{id_str}_Results_run_id_") and file.endswith(".csv"):
                files.append(file)
            elif file.startswith(f"{db_name}_Configuration_{id_str}_Network") and file.endswith(".txt"):
                network_files.append(file)

        # check that there are 10 files
        if len(files) != 10:
            print(f"Id: {id} has {len(files)} files")
            continue
        for i, file in enumerate(files):
            df = pd.read_csv(f"../TEMP/Train/{db_name}/Results/{file}", delimiter=";")
            df['FileId'] = id_str
            # concatenate the dataframes
            if df_all is None:
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

    # x values
    cmap = matplotlib.cm.get_cmap('tab20_r')
    # get normalized colors
    color_vals = [i/len(ids) for i in range(len(ids))]
    colors = [cmap(val) for val in color_vals]
    x = [i for i in range(1, 51)]
    ys = []
    stds = []

    # group by file id
    groups = df_all.groupby('FileId')
    # for each group group by epoch and get the mean and std
    for i, id in enumerate(ids):
        id_str = str(id).zfill(6)
        group = groups.get_group(id_str).copy()
        group['EpochAccuracy'] = group['EpochAccuracy'] * group['TrainingSize']
        group['EpochLoss'] *= group['TrainingSize']
        group['ValidationAccuracy'] *= group['ValidationSize']
        group['ValidationLoss'] *= group['ValidationSize']
        group['TestAccuracy'] *= group['TestSize']
        # f = lambda x: sum(x['TrainingSize'] * x['EpochAccuracy']) / sum(x['TrainingSize'])

        group_mean = group.groupby('Epoch').mean(numeric_only=True)
        group_mean['EpochAccuracy'] /= group_mean['TrainingSize']
        group_mean['EpochLoss'] /= group_mean['TrainingSize']
        group_mean['ValidationAccuracy'] /= group_mean['ValidationSize']
        group_mean['ValidationLoss'] /= group_mean['ValidationSize']
        group_mean['TestAccuracy'] /= group_mean['TestSize']
        group_std = group.groupby('Epoch').std(numeric_only=True)
        group_std['EpochAccuracy'] /= group_mean['TrainingSize']
        group_std['EpochLoss'] /= group_mean['TrainingSize']
        group_std['ValidationAccuracy'] /= group_mean['ValidationSize']
        group_std['ValidationLoss'] /= group_mean['ValidationSize']
        group_std['TestAccuracy'] /= group_mean['TestSize']
        # EpochAccuracy as list
        accuracies = group_mean[y_val].tolist()
        dev = group_std[y_val].tolist()
        ys.append(accuracies)
        stds.append(dev)
        # plot the EpochAccuracy vs Epoch
        #if i == 0:
        #    ax = group_mean.plot(y=y_val, yerr=group_std[y_val], label=id_str, color=['blue', 'blue', 'green'])
        #else:
        #    group_mean.plot(y=y_val, yerr=group_std[y_val], ax=ax, label=id_str, color=['blue'])

    for i,y in enumerate(ys):
        # y_upper
        y_upper = [y_val + std for y_val, std in zip(ys[i], stds[i])]
        # y_lower
        y_lower = [y_val - std for y_val, std in zip(ys[i], stds[i])]
        #plt.fill_between(x, y_lower, y_upper, color=colors[i], alpha=0.1)
        plt.errorbar(x, y, stds[i], color=colors[i], label=ids[i], alpha=0.5)
        plt.plot(x, y, color=colors[i])
    # save to tikz
    # tikzplotlib.save(f"Results/{db_name}/Plots/{y_val}.tex")
    # set the title
    # two columns for the legend
    plt.legend(ncol=4)
    plt.title(f"{db_name}, {y_val}")
    # set x-axis title to Epoch
    plt.xlabel("Epoch")
    # set y-axis title to accuracy
    plt.ylabel("Accuracy (%)")
    # set y-axis from 0 to 100
    plt.ylim(0, 100)
    plt.savefig(f"../TEMP/Train/{db_name}/Plots/{y_val}.svg")
    plt.show()


def main():
    # plot the training and validation accuracy for the given database
    db_name = 'DHFR'
    epoch_accuracy(db_name, 'Validation')


if __name__ == '__main__':
    main()