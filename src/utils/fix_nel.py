# open all _Labels.txt files and add the graph dataset name in the first column
from pathlib import Path


def fix(path: Path):
    for file in path.rglob("*_Labels.txt"):
        with open(file, "r") as f:
            lines = f.readlines()
            # get dataset name from the path
            dataset_name = file.parts[-3]
            # add dataset name to the first column
            lines = [f"{dataset_name} {line}" for line in lines]
        with open(file, "w") as f:
            f.writelines(lines)
    # there is no file in the root directory print which path was searched
    if not list(path.rglob("*_Labels.txt")):
        print(f"No files found in {path}")

if __name__ == "__main__":
    #fix(Path("Reproduce/Data/SyntheticDatasets/"))
    fix(Path("Testing/RealWorldGraphs/ZINC/"))