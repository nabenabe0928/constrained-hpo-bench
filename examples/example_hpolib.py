import os

from chpobench import HPOLib


bench = HPOLib(
    data_path=os.path.join(os.environ["HOME"], "hpo_benchmarks/hpolib/"),
    dataset_name="naval_propulsion",
    quantiles={"runtime": 0.1, "model_size": 0.5},
    metric_names=["runtime", "loss", "model_size"],
)

config = {
    config_info.name: config_info.choices[0]
    if hasattr(config_info, "choices")
    else config_info.seq[0]
    for config_info in bench.config_space
}

print(bench.constraints)
print(bench.constraint_info)
print(bench(config))
