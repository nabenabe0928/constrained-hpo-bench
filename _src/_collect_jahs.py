import itertools
import json
import os
from argparse import ArgumentParser
from dataclasses import dataclass

import pandas as pd

from _src._jahs_wrapper import BenchmarkWrapper


@dataclass(frozen=True)
class _ObjectiveNames:
    loss: str = "valid-acc"
    model_size: str = "size_MB"
    runtime: str = "runtime"


TARGET_DIR = "_src/jahs_grid_data"
DATASET_NAMES = ["colorectal_histology", "cifar10", "fashion_mnist"]
OBJ_NAMES = _ObjectiveNames()
SEARCH_SPACE = json.load(open("chpobench/discrete_spaces.json"))["jahs-bench-201"]
FIXED_CONFIGS = {p: [] for p in SEARCH_SPACE}
for ps in itertools.product(*(vs for vs in SEARCH_SPACE.values())):
    for name, val in zip(SEARCH_SPACE, ps):
        FIXED_CONFIGS[name].append(val)

FIXED_CONFIGS["Optimizer"] = "SGD"
FIXED_CONFIGS["Resolution"] = 1.0
FIXED_CONFIGS["LearningRate"] = 0.0  # Dummy value for init.
FIXED_CONFIGS["WeightDecay"] = 0.0  # Dummy value for init.


def save_results(dataset_name: str) -> None:
    os.makedirs(TARGET_DIR, exist_ok=True)
    bench = BenchmarkWrapper(task=dataset_name)
    config_table = pd.DataFrame(FIXED_CONFIGS)
    results = []

    for ps in itertools.product(
        *(
            [0.001, 0.01, 0.1, 1.0],  # LearningRate
            [0.00001, 0.0001, 0.001, 0.01],  # WeightDecay
        )
    ):
        print(ps)
        config_table["LearningRate"] = ps[0]
        config_table["WeightDecay"] = ps[1]
        preds = bench(config_table)
        preds["valid-err"] = 100.0 - preds["valid-acc"]
        preds = preds.drop(columns=["valid-acc"])
        results.append(preds)

    df = pd.concat(results, ignore_index=True).astype("float32")
    df["valid-err"] = (df["valid-err"] * 100).astype(int) / 100
    df["runtime"] = df["runtime"].astype(int) / 1.0
    df["size_MB"] = (df["size_MB"] * 10**5).astype(int) / 10**5
    df.to_csv(
        os.path.join(TARGET_DIR, f"{dataset_name}.csv"),
        index=False,
    )


if __name__ == "__main__":
    # NOTE: For Fashion-MNIST, we need to change `sklearn/utils/_encode.py` manually.
    # Before: if np.isnan(known_values).any():
    # After: if False and np.isnan(known_values).any():
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="cifar10")
    args = parser.parse_args()
    dataset_name = args.dataset
    save_results(dataset_name)
