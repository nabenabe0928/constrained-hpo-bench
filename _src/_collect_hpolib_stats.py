from __future__ import annotations

import itertools
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np

import pandas as pd

from tqdm import tqdm

from chpobench.constants import QUANTILES


@dataclass(frozen=True)
class _ObjectiveNames:
    loss: str = "valid_mse"
    size: str = "n_params"
    runtime: str = "runtime"


N_SEEDS = 4
OBJ_NAMES = _ObjectiveNames()
SEARCH_SPACE = json.load(open("chpobench/search_spaces.json"))["hpolib"]
N_TOTAL = np.prod([len(vs) for vs in SEARCH_SPACE.values()])
DATASET_NAMES = [
    "parkinsons_telemonitoring.pkl",
    "protein_structure.pkl",
    "naval_propulsion.pkl",
    "slice_localization.pkl",
]


def get_dataframe(data_path: str) -> pd.DataFrame:
    data = pickle.load(open(data_path, mode="rb"))
    obj_vals = {param_name: [] for param_name in OBJ_NAMES.__dict__.values()}
    for vs in tqdm(itertools.product(*list(SEARCH_SPACE.values())), total=N_TOTAL):
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


def get_thresholds(df: pd.DataFrame) -> dict[str, list[float]]:
    thresholds = {}
    indices = [int(N_SEEDS * N_TOTAL * q) - 1 for q in QUANTILES]

    for param_name in list(OBJ_NAMES.__dict__.values())[1:]:
        sorted_vals = np.sort(df[param_name].to_numpy())
        thresholds[param_name] = sorted_vals[indices].tolist()

    return thresholds


def get_mask(df: pd.DataFrame, threshold: dict[str, float]) -> list[bool]:
    mask = True
    for k, v in threshold.items():
        mask = mask & (df[k] <= v)

    return mask


def get_feasible_ratio(df: pd.DataFrame, threshold: dict[str, float]) -> float:
    mask = get_mask(df, threshold)
    return np.sum(mask) / len(mask)


def get_optimal_values(df: pd.DataFrame, threshold: dict[str, float]) -> float:
    mask = get_mask(df, threshold)
    return df[mask][OBJ_NAMES.loss].min()


def get_overlap_ratio_with_top_configs(
    df: pd.DataFrame,
    threshold: dict[str, float],
    top_val: float,
) -> float:
    mask = get_mask(df, threshold)
    return np.sum(df[mask][OBJ_NAMES.loss] <= top_val) / len(mask)


def create_database(data_path: str) -> pd.DataFrame:
    df = get_dataframe(data_path)
    thresholds = get_thresholds(df)
    top_vals = np.sort(df[OBJ_NAMES.loss].to_numpy())[
        [
            int(N_TOTAL * N_SEEDS * 0.1),
            int(N_TOTAL * N_SEEDS * 0.01),
        ]
    ]

    data = {
        f"{OBJ_NAMES.size}_quantile": [],
        f"{OBJ_NAMES.size}_threshold": [],
        f"{OBJ_NAMES.runtime}_quantile": [],
        f"{OBJ_NAMES.runtime}_threshold": [],
        "optimal_val": [],
        "feasible_ratio": [],
        "top_10%_overlap": [],
        "top_1%_overlap": [],
    }
    quantiles = QUANTILES[:]
    for qs, ts in zip(quantiles, thresholds[OBJ_NAMES.size]):
        for qr, tr in zip(quantiles, thresholds[OBJ_NAMES.runtime]):
            th = {OBJ_NAMES.size: ts, OBJ_NAMES.runtime: tr}
            th = {k: v for k, v in th.items() if v is not None}
            if len(th) == 0:
                continue

            data[f"{OBJ_NAMES.size}_quantile"].append(qs)
            data[f"{OBJ_NAMES.size}_threshold"].append(ts)
            data[f"{OBJ_NAMES.runtime}_quantile"].append(qr)
            data[f"{OBJ_NAMES.runtime}_threshold"].append(tr)
            data["optimal_val"].append(get_optimal_values(df, threshold=th))
            data["feasible_ratio"].append(get_feasible_ratio(df, threshold=th))
            data["top_10%_overlap"].append(
                get_overlap_ratio_with_top_configs(df, th, top_vals[0])
            )
            data["top_1%_overlap"].append(
                get_overlap_ratio_with_top_configs(df, th, top_vals[1])
            )

    return pd.DataFrame(data)


if __name__ == "__main__":
    target_path = "chpobench/metadata/"
    os.makedirs(target_path, exist_ok=True)
    for dataset_name in DATASET_NAMES:
        data_path = os.path.join(os.environ["HOME"], f"tabular_benchmarks/hpolib/{dataset_name}")
        db = create_database(data_path)
        db.to_csv(os.path.join(target_path, f"{dataset_name[:-4]}.csv"))
