import os

from chpobench import HPOLib


bench = HPOLib(
    data_path=os.path.join(os.environ["HOME"], "hpo_benchmarks/hpolib/"),
    dataset_name="naval_propulsion",
    quantiles={"runtime": 0.1, "model_size": 0.5},
    metric_names=["runtime", "loss", "model_size"],
)

config = {
    "activation_fn_1": "relu",
    "activation_fn_2": "relu",
    "batch_size": 8,
    "dropout_1": 0.3,
    "dropout_2": 0.3,
    "init_lr": 5e-4,
    "lr_schedule": "cosine",
    "n_units_1": 32,
    "n_units_2": 32,
}

print(bench.constraints)
print(bench(config))
