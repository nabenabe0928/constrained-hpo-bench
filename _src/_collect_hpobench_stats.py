from __future__ import annotations

import itertools
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np

import pandas as pd

from _src._collector import Collector, ObjectiveNames


@dataclass(frozen=True)
class HPOLibObjectiveNames(ObjectiveNames):
    loss: str = "valid_mse"
    model_size: str = "n_params"
    runtime: str = "runtime"


DATASET_NAMES = [
    "australian.pkl",
    "blood_transfusion.pkl",
    "car.pkl",
    "credit_g.pkl",
    "kc1.pkl",
    "phoneme.pkl",
    "segment.pkl",
    "vehicle.pkl",
]
N_SEEDS = 4
OBJ_NAMES = HPOLibObjectiveNames()
SEARCH_SPACE = json.load(open("chpobench/discrete_spaces.json"))["hpolib"]
N_TOTAL = np.prod([len(vs) for vs in SEARCH_SPACE.values()])


def get_dataframe(data_path: str) -> pd.DataFrame:
    data = pickle.load(open(data_path, mode="rb"))
    obj_vals = {param_name: [] for param_name in OBJ_NAMES.__dict__.values()}
    for vs in itertools.product(*list(SEARCH_SPACE.values())):
        config = {k: v for k, v in zip(SEARCH_SPACE, vs)}
        query = data[json.dumps(config)]
        for obj_name, vals in obj_vals.items():
            vals.extend(
                [
                    query[obj_name][seed][99]
                    if obj_name == OBJ_NAMES.loss
                    else query[obj_name][seed]
                    for seed in range(N_SEEDS)
                ]
            )

    return pd.DataFrame(obj_vals)


if __name__ == "__main__":
    target_path = "chpobench/metadata/"
    os.makedirs(target_path, exist_ok=True)
    for dataset_name in DATASET_NAMES:
        data_path = os.path.join(
            os.environ["HOME"], f"hpo_benchmarks/hpolib/{dataset_name}"
        )
        df = get_dataframe(data_path)
        collector = Collector(obj_names=OBJ_NAMES, n_total=N_TOTAL * N_SEEDS)
        db = collector.create_database(df)
        db.to_csv(os.path.join(target_path, f"{dataset_name[:-4]}.csv"), index=False)
