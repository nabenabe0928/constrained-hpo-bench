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
        assert isinstance(value, float)  # mypy redefinition.
        EPS = (self.upper - self.lower) * 1e-5
        return self.lower - EPS <= value <= self.upper + EPS


@dataclass(frozen=True)
class IntDistributionParams(BaseDistributionParams):
    name: str
    lower: int
    upper: int
    log: bool = False

    def __contains__(self, value: int | float | str | bool) -> bool:
        assert isinstance(value, int)  # mypy redefinition.
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
    _curdir = os.path.dirname(os.path.abspath(__file__))

    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        quantiles: dict[str, float],
        metric_names: list[Literal["loss", "model_size", "runtime", "precision", "f1"]] | None = None,
        seed: int | None = None,
    ):
        self._data_path = data_path
        self._dataset_name = dataset_name
        self._validate_dataset_name()
        self._quantiles = quantiles
        self._metric_names = deepcopy(metric_names) if metric_names is not None else self.avail_obj_names
        self._rng = np.random.RandomState(seed)

        if any(q not in constants.QUANTILES for q in quantiles.values()):
            raise ValueError(
                f"`quantiles` for each constraint must be in {constants.QUANTILES}, but got {quantiles}."
            )
        if not set(self._quantiles).issubset(set(self._metric_names)):
            raise ValueError(
                "metric_names must be a superset of the keys specified in quantiles, but got "
                f"{metric_names=} and {quantiles.keys()=}"
            )

        self._init_bench()
        self._constraints: dict[str, float]
        self._set_constraints()
        self._validate_metric_names()

    def _validate_dataset_name(self) -> None:
        if self._dataset_name not in self.dataset_names:
            raise ValueError(
                f"dataset_name must be in {self.dataset_names}, but got {self._dataset_name}."
            )

    def _validate_metric_names(self) -> None:
        avail_obj_names = self.avail_obj_names
        if not set(self._metric_names).issubset(set(avail_obj_names)):
            raise ValueError(
                f"metric_names must be a subset of {avail_obj_names}, but got {self._metric_names}"
            )
        if not set(self._quantiles).issubset(set(avail_obj_names)):
            raise ValueError(
                f"Keys of quantiles must be a subset of {avail_obj_names}, but got {list(self._quantiles.keys())}"
            )

    def _set_constraints(self) -> None:
        constraint_info = self.constraint_info
        mask = True
        quantiles = self._quantiles.copy()
        for cstr_name in self.avail_constraint_names:
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

    def _validate_input(
        self,
        config: dict[str, int | float | str | bool],
        fidels: dict[str, int | float],
    ) -> None:
        config_space = self.config_space
        fidel_space = self.fidel_space
        for name in config:
            if config[name] not in config_space[name]:
                raise ValueError(
                    f"`{name}` must follow {config_space[name]}, but got {config[name]}."
                )
        for name in fidels:
            if fidels[name] not in fidel_space[name]:
                raise ValueError(
                    f"`{name}` must follow {fidel_space[name]}, but got {fidels[name]}."
                )

    @abstractmethod
    def _init_bench(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        config: dict[str, int | float | str | bool],
        fidels: dict[str, int | float] | None,
    ) -> dict[str, float]:
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def config_space(cls) -> dict[str, BaseDistributionParams]:
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def fidel_space(cls) -> dict[str, BaseDistributionParams]:
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def avail_obj_names(cls) -> list[str]:
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def avail_constraint_names(cls) -> list[str]:
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def dataset_names(cls) -> list[str]:
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def discrete_space(cls) -> dict[str, list[int | float | str | bool]]:
        raise NotImplementedError

    @property
    def constraints(self) -> dict[str, float]:
        return self._constraints.copy()

    @property
    def constraint_info(self) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(self._curdir, "metadata", f"{self._dataset_name}.csv")
        )

    @classmethod
    def avail_quantiles(cls) -> list[float]:
        return deepcopy(constants.QUANTILES)

    @classmethod
    def get_constraint_info(cls, dataset_name: str) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(cls._curdir, "metadata", f"{dataset_name}.csv")
        )
