from __future__ import annotations

from typing import Final


_QUANTILES: Final[list[float]] = [
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    1.0,
]
_EPOCHS_KEY: Final[str] = "epochs"
_RUNTIME_KEY: Final[str] = "runtime"
_MODEL_SIZE_KEY: Final[str] = "model_size"
_LOSS_KEY: Final[str] = "loss"
_PRECISION_KEY: Final[str] = "precision"
_F1_KEY: Final[str] = "f1"
