from __future__ import annotations

import json
import os
import pickle

from chpobench.base import (
    BaseBench,
    BaseDistributionParams,
    CategoricalDistributionParams,
    IntDistributionParams,
    OrdinalDistributionParams,
)


class HPOLib(BaseBench):
    def _init_bench(self) -> None:
        self._dataset_names = [
            "parkinsons_telemonitoring",
            "protein_structure",
            "naval_propulsion",
            "slice_localization",
        ]
        self._validate_dataset_name()
        self._search_space = json.load(
            open(os.path.join(self._curdir, "discrete_spaces.json"))
        )["hpolib"]
        self._data = pickle.load(
            open(os.path.join(self._data_path, f"{self._dataset_name}.pkl"), mode="rb")
        )
        self._avail_constraint_names = ["model_size", "runtime"]
        self._avail_obj_names = ["model_size", "runtime", "loss"]

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
                    for key, choices in self._search_space.items()
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

    @property
    def config_space(self) -> dict[str, BaseDistributionParams]:
        config_space: dict[str, BaseDistributionParams] = {}
        for name, choices in self._search_space.items():
            if isinstance(choices[0], str):
                config_space[name] = CategoricalDistributionParams(
                    name=name, choices=choices
                )
            else:
                config_space[name] = OrdinalDistributionParams(name=name, seq=choices)

        return config_space

    @property
    def fidel_space(self) -> dict[str, BaseDistributionParams]:
        return {"epochs": IntDistributionParams(name="epochs", lower=1, upper=100)}
