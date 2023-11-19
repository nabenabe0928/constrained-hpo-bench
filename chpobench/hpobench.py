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


class HPOBench(BaseBench):
    def _init_bench(self):
        self._dataset_names = [
            "australian",
            "blood_transfusion",
            "car",
            "credit_g",
            "kc1",
            "phoneme",
            "segment",
            "vehicle",
        ]
        self._validate_dataset_name()
        self._search_space = json.load(
            open(os.path.join(self._curdir, "discrete_spaces.json"))
        )["hpobench"]
        self._data = pickle.load(
            open(os.path.join(self._data_path, f"{self._dataset_name}.pkl"), mode="rb")
        )
        self._avail_constraint_names = ["precision", "runtime"]
        self._avail_obj_names = ["precision", "f1", "runtime", "loss"]

    def __call__(
        self,
        config: dict[str, int | float | str | bool],
        fidels: dict[str, int | float] | None = None,
    ) -> dict[str, float]:
        EPOCH_CHOICES, N_SEEDS = [3, 9, 27, 81, 243], 5
        fidels = {} if fidels is None else fidels.copy()
        self._validate_input(config, fidels)
        epochs = fidels.get("epochs", EPOCH_CHOICES[-1])
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
            raise KeyError(f"HPOBench does not have the config: {config}")

        if epochs not in EPOCH_CHOICES:
            raise ValueError(
                f"`epochs` of HPOLib must be in {EPOCH_CHOICES}, but got {epochs=}"
            )

        results = dict(
            loss=1.0 - query["bal_acc"][seed][epochs],
            runtime=query["runtime"][seed][epochs],
            f1=query["f1"][seed][epochs],
            precision=query["precision"][seed][epochs],
        )
        return {k: v for k, v in results.items() if k in self._metric_names}

    @property
    def config_space(self) -> dict[str, BaseDistributionParams]:
        config_space = {
            name: OrdinalDistributionParams(name=name, seq=choices)
            for name, choices in self._search_space.items()
        }

        return config_space

    @property
    def fidel_space(self) -> list[BaseDistributionParams]:
        return {
            "epochs": OrdinalDistributionParams(name="epochs", seq=[3, 9, 27, 81, 243])
        }
