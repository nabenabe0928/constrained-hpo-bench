from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

import pandas as pd

from chpobench import constants


class BaseDistributionParams:
    pass


@dataclass(frozen=True)
class FloatDistributionParams(BaseDistributionParams):
    name: str
    lower: float
    upper: float
    log: bool = False


@dataclass(frozen=True)
class IntDistributionParams(BaseDistributionParams):
    name: str
    lower: int
    upper: int
    log: bool = False


@dataclass(frozen=True)
class OrdinalDistributionParams(BaseDistributionParams):
    name: str
    seq: list[int | float]


@dataclass(frozen=True)
class CategoricalDistributionParams(BaseDistributionParams):
    name: str
    choices: list[str | bool]


class BaseBench(metaclass=ABCMeta):
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        quantiles: dict[str, float],
        seed: int | None = None,
    ):
        self._data_path = data_path
        self._dataset_name = dataset_name
        self._quantiles = quantiles
        self._rng = np.random.RandomState(seed)
        self._curdir = os.path.dirname(os.path.abspath(__file__))

        if any(q not in constants.QUANTILES for q in quantiles.values()):
            raise ValueError(
                f"`quantiles` for each constraint must be in {constants.QUANTILES}, but got {quantiles}."
            )

        self._constraints: dict[str, float]
        self._set_constraints()
        self._init_bench()

    def _set_constraints(self) -> None:
        constraint_info = pd.read_csv(
            os.path.join(self._curdir, "metadata", f"{self._dataset_name}.csv")
        )
        quantiles = self._quantiles.copy()
        if "model_size" not in quantiles:
            quantiles["model_size"] = 1.0
        if "runtime" not in quantiles:
            quantiles["runtime"] = 1.0

        cond_model_size = (
            constraint_info["model_size_quantile"] == quantiles["model_size"]
        )
        cond_runtime = constraint_info["runtime_quantile"] == quantiles["runtime"]
        if np.sum(cond_model_size & cond_runtime) != 1:
            raise ValueError(f"`{quantiles=}` was not correctly specified.")

        target = constraint_info[cond_model_size & cond_runtime]
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
    def config_space(self) -> list[BaseDistributionParams]:
        raise NotImplementedError

    @property
    @abstractmethod
    def fidel_space(self) -> list[BaseDistributionParams]:
        raise NotImplementedError

    @property
    def constraints(self) -> dict[str, float]:
        return self._constraints.copy()
