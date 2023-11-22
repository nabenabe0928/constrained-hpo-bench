from __future__ import annotations

import numpy as np

import pandas as pd

from chpobench.constants import _QUANTILES


class ObjectiveNames:
    loss: str
    runtime: str
    model_size: str | None = None
    precision: str | None = None
    f1: str | None = None


class Collector:
    def __init__(self, obj_names: ObjectiveNames, n_total: int):
        self._obj_names = obj_names
        self._n_total = n_total

    def get_thresholds(self, df: pd.DataFrame) -> dict[str, list[float]]:
        thresholds = {}
        indices = [int(self._n_total * q) - 1 for q in _QUANTILES]

        for param_name in ["loss", "runtime", "model_size", "precision"]:
            obj_name = getattr(self._obj_names, param_name)
            if obj_name is None:
                continue

            sorted_vals = np.sort(df[obj_name].to_numpy())
            sorted_vals = (
                sorted_vals if param_name != "precision" else sorted_vals[::-1]
            )
            thresholds[obj_name] = sorted_vals[indices].tolist()

        return thresholds

    @staticmethod
    def get_mask(df: pd.DataFrame, threshold: dict[str, float]) -> list[bool]:
        mask = True
        for k, v in threshold.items():
            if k != "precision":
                mask = mask & (df[k] <= v)
            else:
                mask = mask & (df[k] >= v)

        return mask

    @classmethod
    def get_feasible_ratio(cls, df: pd.DataFrame, threshold: dict[str, float]) -> float:
        mask = cls.get_mask(df, threshold)
        return np.sum(mask) / len(mask)

    def get_optimal_values(
        self, df: pd.DataFrame, threshold: dict[str, float]
    ) -> float:
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

        first_cstr = (
            "model_size" if self._obj_names.model_size is not None else "precision"
        )

        data = {
            f"{first_cstr}_quantile": [],
            f"{first_cstr}_threshold": [],
            "runtime_quantile": [],
            "runtime_threshold": [],
            "optimal_val": [],
            "feasible_ratio": [],
            "top_10%_overlap": [],
            "top_1%_overlap": [],
        }
        quantiles = _QUANTILES[:]
        for q1, t1 in zip(quantiles, thresholds[getattr(self._obj_names, first_cstr)]):
            for q2, t2 in zip(quantiles, thresholds[self._obj_names.runtime]):
                th = {
                    getattr(self._obj_names, first_cstr): t1,
                    self._obj_names.runtime: t2,
                }
                if len(th) == 0:
                    continue

                data[f"{first_cstr}_quantile"].append(q1)
                data[f"{first_cstr}_threshold"].append(t1)
                data["runtime_quantile"].append(q2)
                data["runtime_threshold"].append(t2)
                data["optimal_val"].append(self.get_optimal_values(df, threshold=th))
                data["feasible_ratio"].append(self.get_feasible_ratio(df, threshold=th))
                data["top_10%_overlap"].append(
                    self.get_overlap_ratio_with_top_configs(df, th, top_vals[0])
                )
                data["top_1%_overlap"].append(
                    self.get_overlap_ratio_with_top_configs(df, th, top_vals[1])
                )

        return pd.DataFrame(data)
