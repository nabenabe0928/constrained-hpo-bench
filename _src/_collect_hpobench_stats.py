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
class HPOBenchObjectiveNames(ObjectiveNames):
    loss: str = "err"
    runtime: str = "runtime"
    precision: str = "precision"
    f1: str = "f1"


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
N_SEEDS = 5
OBJ_NAMES = HPOBenchObjectiveNames()
SEARCH_SPACE = json.load(open("chpobench/discrete_spaces.json"))["hpobench"]
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
            [1.0 - query["bal_acc"][seed][243] for seed in range(N_SEEDS)]
        )
        obj_vals[OBJ_NAMES.precision].extend(
            [query[OBJ_NAMES.precision][seed][243] for seed in range(N_SEEDS)]
        )
        obj_vals[OBJ_NAMES.f1].extend(
            [query[OBJ_NAMES.f1][seed][243] for seed in range(N_SEEDS)]
        )
        obj_vals[OBJ_NAMES.runtime].extend(
            [query[OBJ_NAMES.runtime][seed][243] for seed in range(N_SEEDS)]
        )

    return pd.DataFrame(obj_vals)


if __name__ == "__main__":
    target_path = "chpobench/metadata/"
    os.makedirs(target_path, exist_ok=True)
    for dataset_name in DATASET_NAMES:
        data_path = os.path.join(
            os.environ["HOME"], f"hpo_benchmarks/hpobench/{dataset_name}"
        )
        df = get_dataframe(data_path)
        collector = Collector(obj_names=OBJ_NAMES, n_total=N_TOTAL * N_SEEDS)
        db = collector.create_database(df)
        db.to_csv(os.path.join(target_path, f"{dataset_name[:-4]}.csv"), index=False)
