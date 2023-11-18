from __future__ import annotations

from typing import Iterable, Literal
import os

import pandas as pd

from jahs_bench import Benchmark


class BenchmarkWrapper(Benchmark):
    def __init__(
        self,
        save_dir: str = DATA_DIR,
        task: Literal["colorectal_histology", "cifar10", "fashion_mnist"] = "cifar10",
        download: bool = False,
        metrics: list[str] | None = None,
    ):
        metrics = ["valid-acc", "size_MB", "runtime"] if metrics is None else metrics[:]
        super().__init__(
            task=task, download=download, save_dir=save_dir, metrics=metrics
        )

    def __call__(
        self,
        feats: pd.DataFrame,
        nepochs: Optional[int] = 200,
    ) -> pd.DataFrame:

        assert nepochs > 0
        feats.loc[:, "epoch"] = nepochs

        outputs = []
        for model in self._surrogates.values():
            outputs.append(model.predict(feats))

        return pd.concat(outputs, axis=1)
