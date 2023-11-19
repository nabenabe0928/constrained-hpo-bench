from __future__ import annotations

import json
import os
import pickle
from copy import deepcopy

from chpobench.base import (
    BaseBench,
    BaseDistributionParams,
    CategoricalDistributionParams,
    IntDistributionParams,
    OrdinalDistributionParams,
)


class HPOLib(BaseBench):
    _discrete_space: dict[str, list[int | float | bool | str]] = json.load(
        open(os.path.join(BaseBench._curdir, "discrete_spaces.json"))
    )["hpolib"]

    def _init_bench(self) -> None:
        self._data = pickle.load(
            open(os.path.join(self._data_path, f"{self._dataset_name}.pkl"), mode="rb")
        )

    def __call__(
        self,
        config: dict[str, int | float | str | bool],
        fidels: dict[str, int | float] | None = None,
    ) -> dict[str, float]:
        MAX_EPOCHS, N_SEEDS = 100, 4
        fidels = {} if fidels is None else fidels.copy()
        self._validate_input(config, fidels)
        epochs = fidels.get("epochs", MAX_EPOCHS)
        seed = self._rng.randint(N_SEEDS)
        try:
            index = "".join(
                [
                    str(choices.index(config[key]))
                    for key, choices in self.discrete_space.items()
                ]
            )
            query = self._data[index]
        except KeyError:
            raise KeyError(f"HPOLib does not have the config: {config}")

        if epochs > MAX_EPOCHS or epochs < 1:
            raise ValueError(
                f"`epochs` of HPOLib must be in [1, {MAX_EPOCHS}], but got {epochs=}"
            )

        results = dict(
            loss=query["valid_mse"][seed][epochs],
            model_size=query["n_params"],
            runtime=query["runtime"][seed] * epochs / MAX_EPOCHS,
        )
        return {k: v for k, v in results.items() if k in self._metric_names}

    @classmethod
    @property
    def dataset_names(cls) -> list[str]:
        return [
            "parkinsons_telemonitoring",
            "protein_structure",
            "naval_propulsion",
            "slice_localization",
        ]

    @classmethod
    @property
    def avail_obj_names(cls) -> list[str]:
        return ["model_size", "runtime", "loss"]

    @classmethod
    @property
    def avail_constraint_names(cls) -> list[str]:
        return ["model_size", "runtime"]

    @classmethod
    @property
    def config_space(cls) -> dict[str, BaseDistributionParams]:
        config_space: dict[str, BaseDistributionParams] = {}
        for name, choices in cls.discrete_space.items():
            if isinstance(choices[0], str):
                config_space[name] = CategoricalDistributionParams(
                    name=name, choices=choices
                )
            else:
                config_space[name] = OrdinalDistributionParams(name=name, seq=choices)

        return config_space

    @classmethod
    @property
    def fidel_space(cls) -> dict[str, BaseDistributionParams]:
        return {"epochs": IntDistributionParams(name="epochs", lower=1, upper=100)}

    @classmethod
    @property
    def discrete_space(cls) -> dict[str, list[int | float | str | bool]]:
        return deepcopy(cls._discrete_space)
