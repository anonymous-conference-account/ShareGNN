# Share GNNs

## Setting up the Environment

1. Clone the repository

2. Install the required packages using the install.sh script:
   ```bash
   # Make the script executable (this is necessary before you can run it)
   chmod +x install.sh
   # Run the installation script
   ./install.sh
   ```

   > **Note:** The `chmod +x install.sh` command makes the script executable. This is a necessary step on Unix-based systems (Linux/macOS) before you can run a shell script. If you're on Windows using Git Bash or WSL, you'll also need this command.

   This script will:
   - Check if Python 3.12 is installed (with installation hints if it's not)
   - Create a Python virtual environment
   - Install all dependencies from requirements.txt
3. **(for command line)** To run the scripts with the correct paths please set your PYTHONPATH (working directory) to the root directory of the repository.
   ```bash
   export PYTHONPATH=/path/to/ShareGNN
   ```
    **(for IDE)** If you are working in an IDE, you can set the PYTHONPATH in the run configuration. 
   E.g., in PyCharm, you have to change the working directory path to the root directory of the repository.
    Go to ```File -> Settings -> Project Structure``` and mark the root directory as ```Sources``` (blue folder icon).


## Reproduce Results
To reproduce the experiments of the paper, simply follow the steps below. All necessary code can be found in the [paper_experiments](paper_experiments) folder.
All experiments run on an AMD Ryzen 9 7950X with 16 cores and 32 threads and 128 GB of RAM.

First navigate to the paper_experiments folder:

```bash
cd paper_experiments
```

### Substructure Counting
```bash
chmod +x substructure_counting.sh & ./substructure_counting.sh
```
### Synthetic Datasets
```bash
chmod +x synthetic.sh & ./synthetic.sh
```

### Real World Classification
```bash
chmod +x TUDatasets.sh & ./TUDatasets.sh
```

### ZINC (12k and 250k)
```bash
chmod +x ZINC.sh & ./ZINC.sh
```
```bash
chmod +x ZINC_full.sh & ./ZINC_full.sh
```



The following steps are executed:

   - download of the datasets
   - preprocessing of the datasets
   - experiments regarding fair evaluation, the standard evaluation, the synthetic data, the baselines and the ablation experiments
   - grid search to find the best hyperparameters for different models
   - best models three times with different seeds
   - evaluation of the results

All results will be saved in the [paper_experiments/Results](paper_experiments/Results) folder.

For each experiment and each dataset the following evaluation files are produced:

- ```summary.csv```: contains the results of the grid search (fair evaluation) one row per hyperparameter setting
- ```summary_best.csv```: contains the results of the best model (hyperparameter setting) one row per seed
- ```summary_best_mean.csv```: contains the mean and standard deviation of the best model results over all seeds
