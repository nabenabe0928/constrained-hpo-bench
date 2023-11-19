from __future__ import annotations

import json
import os
from copy import deepcopy

from jahs_bench import Benchmark

from chpobench.base import (
    BaseBench,
    BaseDistributionParams,
    CategoricalDistributionParams,
    FloatDistributionParams,
    IntDistributionParams,
    OrdinalDistributionParams,
)


class JAHSBench201(BaseBench):
    _discrete_space: dict[str, list[int | float | bool | str]] = json.load(
        open(os.path.join(BaseBench._curdir, "discrete_spaces.json"))
    )["jahs-bench-201"]

    def _init_bench(self) -> None:
        metric_dict = {
            "loss": "valid-acc",
            "runtime": "runtime",
            "model_size": "size_MB",
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
        epochs = fidels.get("epochs", 200)
        resol = fidels.get("Resolution", 1.0)
        config["Optimizer"] = "SGD"
        config["Resolution"] = resol

        preds = self._surrogate(config, nepochs=epochs)[epochs]
        metric_dict = {
            "valid-acc": "loss",
            "runtime": "runtime",
            "size_MB": "model_size",
        }
        return {
            metric_dict[k]: 100.0 - v if k == "valid-acc" else v
            for k, v in preds.items()
        }

    @classmethod
    @property
    def dataset_names(cls) -> list[str]:
        return ["colorectal_histology", "cifar10", "fashion_mnist"]

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
            "epochs": IntDistributionParams(name="epochs", lower=1, upper=100),
            "Resolution": FloatDistributionParams(
                name="Resolution", lower=0.0, upper=1.0
            ),
        }

    @classmethod
    @property
    def discrete_space(cls) -> dict[str, list[int | float | str | bool]]:
        return deepcopy(cls._discrete_space)
