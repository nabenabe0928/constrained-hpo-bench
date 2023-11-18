from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np


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
        dataset_name: str,
        quantiles: dict[str, float],
        seed: int | None = None,
    ):
        self._dataset_name = dataset_name
        self._quantiles = quantiles
        self._rng = np.random.RandomState(seed)

    @abstractmethod
    def _init_bench(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, config: dict[str, int | float | str | bool], fidels: dict[str, int | float]) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def config_space(self) -> dict[str, BaseDistributionParams]:
        raise NotImplementedError
