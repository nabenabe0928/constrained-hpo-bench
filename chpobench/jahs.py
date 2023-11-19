from __future__ import annotations

import json
import os
import pickle

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
    def _init_bench(self):
        self._dataset_names = ["colorectal_histology", "cifar10", "fashion_mnist"]
        self._validate_dataset_name()
        self._discrete_space = json.load(
            open(os.path.join(self._curdir, "discrete_spaces.json"))
        )["jahs-bench-201"]
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
        return {metric_dict[k]: 100.0 - v if k == "valid-acc" else v for k, v in preds.items()}

    @property
    def config_space(self) -> list[BaseDistributionParams]:
        config_space = [
            FloatDistributionParams(
                name="LearningRate", lower=1e-3, upper=1.0, log=True
            ),
            FloatDistributionParams(
                name="WeightDecay", lower=1e-5, upper=1e-2, log=True
            ),
        ]
        for name, choices in self._discrete_space.items():
            if name in ["N", "W"]:
                config_space.append(OrdinalDistributionParams(name=name, seq=choices))
            else:
                config_space.append(
                    CategoricalDistributionParams(name=name, choices=choices)
                )

        return config_space

    @property
    def fidel_space(self) -> list[BaseDistributionParams]:
        return [
            IntDistributionParams(name="epochs", lower=1, upper=100),
            FloatDistributionParams(name="Resolution", lower=0.0, upper=1.0),
        ]
