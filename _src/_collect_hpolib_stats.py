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
    runtime: str = "runtime"
    model_size: str | None = "n_params"


DATASET_NAMES = [
    "parkinsons_telemonitoring.pkl",
    "protein_structure.pkl",
    "naval_propulsion.pkl",
    "slice_localization.pkl",
]
N_SEEDS = 4
OBJ_NAMES = HPOLibObjectiveNames()
SEARCH_SPACE = json.load(open("chpobench/discrete_spaces.json"))["hpolib"]
N_TOTAL = np.prod([len(vs) for vs in SEARCH_SPACE.values()])


def get_dataframe(data_path: str) -> pd.DataFrame:
    data = pickle.load(open(data_path, mode="rb"))
    obj_vals = {param_name: [] for param_name in OBJ_NAMES.__dict__.values()}
    for vs in itertools.product(*list(SEARCH_SPACE.values())):
        index = "".join(
            [str(choices.index(v)) for choices, v in zip(SEARCH_SPACE.values(), vs)]
        )
        query = data[index]
        obj_vals[OBJ_NAMES.loss].extend(
            [query[OBJ_NAMES.loss][seed][100] for seed in range(N_SEEDS)]
        )
        obj_vals[OBJ_NAMES.model_size].extend(
            [query[OBJ_NAMES.model_size] for seed in range(N_SEEDS)]
        )
        obj_vals[OBJ_NAMES.runtime].extend(
            [query[OBJ_NAMES.runtime][seed] for seed in range(N_SEEDS)]
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
