from __future__ import annotations

import os

import numpy as np

import pandas as pd

from chpobench.constants import QUANTILES


class ObjectiveNames:
    loss: str
    size: str
    runtime: str


class Collector:
    def __init__(self, obj_names: ObjectiveNames, n_total: int):
        self._obj_names = obj_names
        self._n_total = n_total

    def get_thresholds(self, df: pd.DataFrame) -> dict[str, list[float]]:
        thresholds = {}
        indices = [int(self._n_total * q) - 1 for q in QUANTILES]

        for param_name in list(self._obj_names.__dict__.values())[1:]:
            sorted_vals = np.sort(df[param_name].to_numpy())
            thresholds[param_name] = sorted_vals[indices].tolist()

        return thresholds

    @staticmethod
    def get_mask(df: pd.DataFrame, threshold: dict[str, float]) -> list[bool]:
        mask = True
        for k, v in threshold.items():
            mask = mask & (df[k] <= v)

        return mask

    @classmethod
    def get_feasible_ratio(cls, df: pd.DataFrame, threshold: dict[str, float]) -> float:
        mask = cls.get_mask(df, threshold)
        return np.sum(mask) / len(mask)

    def get_optimal_values(self, df: pd.DataFrame, threshold: dict[str, float]) -> float:
        mask = self.get_mask(df, threshold)
        return df[mask][self._obj_names.loss].min()


    def get_overlap_ratio_with_top_configs(
        self,
        df: pd.DataFrame,
        threshold: dict[str, float],
        top_val: float,
    ) -> float:
        mask = self.get_mask(df, threshold)
        return np.sum(df[mask][self._obj_names.loss] <= top_val) / len(mask)


    def create_database(self, df: pd.DataFrame) -> pd.DataFrame:
        thresholds = self.get_thresholds(df)
        top_vals = np.sort(df[self._obj_names.loss].to_numpy())[
            [
                int(self._n_total * 0.1),
                int(self._n_total * 0.01),
            ]
        ]

        data = {
            f"{self._obj_names.size}_quantile": [],
            f"{self._obj_names.size}_threshold": [],
            f"{self._obj_names.runtime}_quantile": [],
            f"{self._obj_names.runtime}_threshold": [],
            "optimal_val": [],
            "feasible_ratio": [],
            "top_10%_overlap": [],
            "top_1%_overlap": [],
        }
        quantiles = QUANTILES[:]
        for qs, ts in zip(quantiles, thresholds[self._obj_names.size]):
            for qr, tr in zip(quantiles, thresholds[self._obj_names.runtime]):
                th = {self._obj_names.size: ts, self._obj_names.runtime: tr}
                th = {k: v for k, v in th.items() if v is not None}
                if len(th) == 0:
                    continue

                data[f"{self._obj_names.size}_quantile"].append(qs)
                data[f"{self._obj_names.size}_threshold"].append(ts)
                data[f"{self._obj_names.runtime}_quantile"].append(qr)
                data[f"{self._obj_names.runtime}_threshold"].append(tr)
                data["optimal_val"].append(self.get_optimal_values(df, threshold=th))
                data["feasible_ratio"].append(self.get_feasible_ratio(df, threshold=th))
                data["top_10%_overlap"].append(
                    self.get_overlap_ratio_with_top_configs(df, th, top_vals[0])
                )
                data["top_1%_overlap"].append(
                    self.get_overlap_ratio_with_top_configs(df, th, top_vals[1])
                )

        return pd.DataFrame(data)
