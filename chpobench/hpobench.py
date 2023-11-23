from __future__ import annotations

import json
import os
import pickle
from copy import deepcopy
from typing import Final, Literal

from chpobench import constants
from chpobench.base import (
    BaseBench,
    BaseDistributionParams,
    OrdinalDistributionParams,
)


class HPOBench(BaseBench):
    _discrete_space: dict[str, list[int | float | bool | str]] = json.load(
        open(os.path.join(BaseBench._curdir, "discrete_spaces.json"))
    )["hpobench"]
    _EPOCH_CHOICES: Final[list[int]] = [3, 9, 27, 81, 243]
    _N_SEEDS: Final[int] = 5

    def _init_bench(self) -> None:
        self._data = pickle.load(
            open(os.path.join(self._data_path, f"{self._dataset_name}.pkl"), mode="rb")
        )

    def __call__(
        self,
        config: dict[str, int | float | str | bool],
        fidels: dict[str, int | float] | None = None,
    ) -> dict[str, float]:
        fidels = {} if fidels is None else fidels.copy()
        self._validate_input(config, fidels)
        epochs = fidels.get(constants._EPOCHS_KEY, self._EPOCH_CHOICES[-1])
        seed = self._rng.randint(self._N_SEEDS)
        try:
            index = "".join(
                [
                    str(choices.index(config[key]))
                    for key, choices in self.discrete_space.items()
                ]
            )
            query = self._data[index]
        except KeyError:
            raise KeyError(f"HPOBench does not have the config: {config}")

        if epochs not in self._EPOCH_CHOICES:
            raise ValueError(
                f"`epochs` of HPOBench must be in {self._EPOCH_CHOICES}, but got {epochs=}"
            )

        results = {
            constants._LOSS_KEY: 1.0 - query["bal_acc"][seed][epochs],
            constants._RUNTIME_KEY: query["runtime"][seed][epochs],
            constants._F1_KEY: query["f1"][seed][epochs],
            constants._PRECISION_KEY: query["precision"][seed][epochs],
        }
        return {k: v for k, v in results.items() if k in self._metric_names}

    @classmethod
    @property
    def dataset_names(cls) -> list[str]:
        return [
            "australian",
            "blood_transfusion",
            "car",
            "credit_g",
            "kc1",
            "phoneme",
            "segment",
            "vehicle",
        ]

    @classmethod
    @property
    def avail_obj_names(cls) -> list[str]:
        return [
            constants._LOSS_KEY,
            constants._F1_KEY,
            constants._RUNTIME_KEY,
            constants._PRECISION_KEY,
        ]

    @classmethod
    @property
    def avail_constraint_names(cls) -> list[str]:
        return [constants._RUNTIME_KEY, constants._PRECISION_KEY]

    @classmethod
    @property
    def directions(cls) -> dict[str, Literal["min", "max"]]:
        return {
            constants._LOSS_KEY: "min",
            constants._F1_KEY: "max",
            constants._RUNTIME_KEY: "min",
            constants._PRECISION_KEY: "max",
        }

    @classmethod
    @property
    def config_space(cls) -> dict[str, BaseDistributionParams]:
        config_space: dict[str, BaseDistributionParams] = {
            name: OrdinalDistributionParams(name=name, seq=choices)
            for name, choices in cls.discrete_space.items()
        }
        return config_space

    @classmethod
    @property
    def fidel_space(cls) -> dict[str, BaseDistributionParams]:
        fidel_space: dict[str, BaseDistributionParams] = {
            constants._EPOCHS_KEY: OrdinalDistributionParams(
                name=constants._EPOCHS_KEY, seq=deepcopy(cls._EPOCH_CHOICES)
            )
        }
        return fidel_space

    @classmethod
    @property
    def discrete_space(cls) -> dict[str, list[int | float | str | bool]]:
        return deepcopy(cls._discrete_space)
