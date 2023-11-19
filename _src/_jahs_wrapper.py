from __future__ import annotations

from typing import Literal
import os

import pandas as pd

from jahs_bench import Benchmark


class BenchmarkWrapper(Benchmark):
    def __init__(
        self,
        task: Literal["colorectal_histology", "cifar10", "fashion_mnist"],
        save_dir: str = os.path.join(os.environ["HOME"], "hpo_benchmarks/jahs/"),
        download: bool = False,
        metrics: list[str] | None = None,
    ):
        metrics = ["valid-acc", "size_MB", "runtime"] if metrics is None else metrics[:]
        super().__init__(
            task=task, download=download, save_dir=save_dir, metrics=metrics
        )

    def __call__(self, feats: pd.DataFrame, nepochs: int | None = 200) -> pd.DataFrame:

        assert nepochs > 0
        feats.loc[:, "epoch"] = nepochs

        outputs = []
        for model in self._surrogates.values():
            outputs.append(model.predict(feats))

        return pd.concat(outputs, axis=1)
