from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Final, Literal

from jahs_bench import Benchmark

from chpobench import constants
from chpobench.base import (
    BaseBench,
    BaseDistributionParams,
    CategoricalDistributionParams,
    FloatDistributionParams,
    IntDistributionParams,
    OrdinalDistributionParams,
)


_RESOL_KEY: Final[str] = "Resolution"
_JAHS_LOSS_KEY: Final[str] = "valid-acc"
_JAHS_RUNTIME_KEY: Final[str] = "runtime"
_JAHS_MODEL_SIZE_KEY: Final[str] = "size_MB"


class JAHSBench201(BaseBench):
    _discrete_space: dict[str, list[int | float | bool | str]] = json.load(
        open(os.path.join(BaseBench._curdir, "discrete_spaces.json"))
    )["jahs-bench-201"]
    _MAX_EPOCHS: Final[int] = 200

    def _init_bench(self) -> None:
        metric_dict = {
            constants._LOSS_KEY: _JAHS_LOSS_KEY,
            constants._RUNTIME_KEY: _JAHS_RUNTIME_KEY,
            constants._MODEL_SIZE_KEY: _JAHS_MODEL_SIZE_KEY,
        }
        self._surrogate = Benchmark(
            task=self._dataset_name,
            save_dir=self._data_path,
            metrics=[metric_dict[name] for name in self._metric_names],
            download=False,
        )

    def __call__(
        self,
        config: dict[str, int | float | str | bool],
        fidels: dict[str, int | float] | None = None,
    ) -> dict[str, float]:
        fidels = {} if fidels is None else fidels.copy()
        self._validate_input(config, fidels)
        epochs = fidels.get(constants._EPOCHS_KEY, self._MAX_EPOCHS)
        resol = fidels.get(_RESOL_KEY, 1.0)
        config["Optimizer"] = "SGD"
        config[_RESOL_KEY] = resol

        preds = self._surrogate(config, nepochs=epochs)[epochs]
        metric_dict = {
            _JAHS_LOSS_KEY: constants._LOSS_KEY,
            _JAHS_RUNTIME_KEY: constants._RUNTIME_KEY,
            _JAHS_MODEL_SIZE_KEY: constants._MODEL_SIZE_KEY,
        }
        return {
            metric_dict[k]: 100.0 - v if k == _JAHS_LOSS_KEY else v
            for k, v in preds.items()
        }

    @classmethod
    @property
    def dataset_names(cls) -> list[str]:
        return ["colorectal_histology", "cifar10", "fashion_mnist"]

    @classmethod
    @property
    def avail_obj_names(cls) -> list[str]:
        return [constants._LOSS_KEY, constants._RUNTIME_KEY, constants._MODEL_SIZE_KEY]

    @classmethod
    @property
    def avail_constraint_names(cls) -> list[str]:
        return [constants._RUNTIME_KEY, constants._MODEL_SIZE_KEY]

    @classmethod
    @property
    def directions(cls) -> dict[str, Literal["min", "max"]]:
        return {
            constants._LOSS_KEY: "min",
            constants._RUNTIME_KEY: "min",
            constants._MODEL_SIZE_KEY: "min",
        }

    @classmethod
    @property
    def config_space(cls) -> dict[str, BaseDistributionParams]:
        config_space: dict[str, BaseDistributionParams] = {
            "LearningRate": FloatDistributionParams(
                name="LearningRate", lower=1e-3, upper=1.0, log=True
            ),
            "WeightDecay": FloatDistributionParams(
                name="WeightDecay", lower=1e-5, upper=1e-2, log=True
            ),
        }
        for name, choices in cls.discrete_space.items():
            if name in ["N", "W"]:
                config_space[name] = OrdinalDistributionParams(name=name, seq=choices)
            else:
                config_space[name] = CategoricalDistributionParams(
                    name=name, choices=choices
                )

        return config_space

    @classmethod
    @property
    def fidel_space(cls) -> dict[str, BaseDistributionParams]:
        return {
            constants._EPOCHS_KEY: IntDistributionParams(
                name=constants._EPOCHS_KEY, lower=1, upper=cls._MAX_EPOCHS
            ),
            _RESOL_KEY: FloatDistributionParams(name=_RESOL_KEY, lower=0.0, upper=1.0),
        }

    @classmethod
    @property
    def discrete_space(cls) -> dict[str, list[int | float | str | bool]]:
        return deepcopy(cls._discrete_space)
