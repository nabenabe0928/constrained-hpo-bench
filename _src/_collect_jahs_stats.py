from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from _src._collector import Collector, ObjectiveNames


@dataclass(frozen=True)
class JAHSObjectiveNames(ObjectiveNames):
    loss: str = "valid-err"
    size: str = "size_MB"
    runtime: str = "runtime"


DATASET_NAMES = ["colorectal_histology.csv", "cifar10.csv", "fashion_mnist.csv"]
OBJ_NAMES = JAHSObjectiveNames()


if __name__ == "__main__":
    target_path = "chpobench/metadata/"
    os.makedirs(target_path, exist_ok=True)
    for dataset_name in DATASET_NAMES:
        print(f"Process {dataset_name}")
        df = pd.read_csv(f"_src/jahs_grid_data/{dataset_name}")
        n_total = len(df)
        collector = Collector(obj_names=OBJ_NAMES, n_total=n_total)
        db = collector.create_database(df)
        db.to_csv(os.path.join(target_path, dataset_name), index=False)
