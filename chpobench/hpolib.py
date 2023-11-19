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
    def _init_bench(self):
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

    def __call__(
        self,
        config: dict[str, int | float | str | bool],
        fidels: dict[str, int | float] | None = None,
    ) -> dict[str, float]:
        epochs = 100 if fidels is None else fidels["epochs"]
        seed = self._rng.randint(4)
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

        MAX_EPOCHS = 100
        if epochs > MAX_EPOCHS or epochs < 1:
            raise ValueError(
                f"`epochs` of HPOLib must be in [1, {MAX_EPOCHS}], but got {epochs=}"
            )

        return dict(
            loss=query["valid_mse"][seed][epochs],
            model_size=query["n_params"],
            runtime=query["runtime"][seed] * epochs / MAX_EPOCHS,
        )

    @property
    def config_space(self) -> list[BaseDistributionParams]:
        config_space = []
        for name, choices in self._search_space.items():
            if isinstance(choices[0], str):
                config_space.append(
                    CategoricalDistributionParams(name=name, choices=choices)
                )
            else:
                config_space.append(OrdinalDistributionParams(name=name, seq=choices))

        return config_space

    @property
    def fidel_space(self) -> list[BaseDistributionParams]:
        return [IntDistributionParams(name="epochs", lower=1, upper=100)]
