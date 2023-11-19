from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

import pandas as pd

from chpobench import constants


class BaseDistributionParams(metaclass=ABCMeta):
    @abstractmethod
    def __contains__(self, value: int | float | str | bool) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class FloatDistributionParams(BaseDistributionParams):
    name: str
    lower: float
    upper: float
    log: bool = False

    def __contains__(self, value: int | float | str | bool) -> bool:
        EPS = (self.upper - self.lower) * 1e-5
        return self.lower - EPS <= value <= self.upper + EPS


@dataclass(frozen=True)
class IntDistributionParams(BaseDistributionParams):
    name: str
    lower: int
    upper: int
    log: bool = False

    def __contains__(self, value: int | float | str | bool) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class OrdinalDistributionParams(BaseDistributionParams):
    name: str
    seq: list[int | float]

    def __contains__(self, value: int | float | str | bool) -> bool:
        return value in self.seq


@dataclass(frozen=True)
class CategoricalDistributionParams(BaseDistributionParams):
    name: str
    choices: list[int | str | bool]

    def __contains__(self, value: int | float | str | bool) -> bool:
        return value in self.choices


class BaseBench(metaclass=ABCMeta):
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        quantiles: dict[str, float],
        metric_names: list[Literal["loss", "model_size", "runtime", "precision", "f1"]],
        seed: int | None = None,
    ):
        self._data_path = data_path
        self._dataset_name = dataset_name
        self._quantiles = quantiles
        self._metric_names = metric_names[:]
        self._rng = np.random.RandomState(seed)
        self._curdir = os.path.dirname(os.path.abspath(__file__))

        if any(q not in constants.QUANTILES for q in quantiles.values()):
            raise ValueError(
                f"`quantiles` for each constraint must be in {constants.QUANTILES}, but got {quantiles}."
            )
        if not set(self._quantiles).issubset(set(self._metric_names)):
            raise ValueError(
                "metric_names must be a superset of the keys specified in quantiles, but got "
                f"{metric_names=} and {quantiles.keys()=}"
            )

        self._avail_constraint_names: list[str]
        self._avail_obj_names: list[str]
        self._dataset_names: list[str]
        self._init_bench()
        self._constraints: dict[str, float]
        self._set_constraints()
        self._validate_metric_names()

    def _validate_dataset_name(self) -> None:
        if self._dataset_name not in self._dataset_names:
            raise ValueError(
                f"dataset_name must be in {self._dataset_names}, but got {self._dataset_name}."
            )

    def _validate_metric_names(self) -> None:
        if not set(self._metric_names).issubset(set(self._avail_obj_names)):
            raise ValueError(
                f"metric_names must be a subset of {self._avail_obj_names}, but got {self._metric_names}"
            )
        if not set(self._quantiles).issubset(set(self._avail_obj_names)):
            raise ValueError(
                f"Keys of quantiles must be a subset of {self._avail_obj_names}, but got {list(self._quantiles.keys())}"
            )

    def _set_constraints(self) -> None:
        constraint_info = self.constraint_info
        mask = True
        quantiles = self._quantiles.copy()
        for cstr_name in self._avail_constraint_names:
            if cstr_name not in quantiles:
                quantiles[cstr_name] = 1.0

            mask = mask & (
                constraint_info[f"{cstr_name}_quantile"] == quantiles[cstr_name]
            )

        if np.sum(mask) != 1:
            raise ValueError(f"`{quantiles=}` was not correctly specified.")
        if constraint_info[mask]["feasible_ratio"].iloc[0] == 0.0:
            raise ValueError(
                "Constraints are too tight. Please loosen some constraint quantiles."
            )

        target = constraint_info[mask]
        self._constraints = {
            key: target[f"{key}_threshold"].iloc[0] for key in self._quantiles
        }

    @abstractmethod
    def _init_bench(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        config: dict[str, int | float | str | bool],
        fidels: dict[str, int | float],
    ) -> dict[str, float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def config_space(self) -> dict[str, BaseDistributionParams]:
        raise NotImplementedError

    @property
    @abstractmethod
    def fidel_space(self) -> dict[str, BaseDistributionParams]:
        raise NotImplementedError

    @property
    def constraints(self) -> dict[str, float]:
        return self._constraints.copy()

    @property
    def constraint_info(self) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(self._curdir, "metadata", f"{self._dataset_name}.csv")
        )
