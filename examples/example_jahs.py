import os

from chpobench import JAHSBench201


bench = JAHSBench201(
    data_path=os.path.join(os.environ["HOME"], "hpo_benchmarks/jahs/"),
    dataset_name="cifar10",
    quantiles={"runtime": 0.1, "model_size": 0.5},
    metric_names=["runtime", "loss", "model_size"],
)

config = {
    config_info.name: config_info.choices[0]
    if hasattr(config_info, "choices")
    else (config_info.seq[0] if hasattr(config_info, "seq") else config_info.lower)
    for config_info in bench.config_space
}

print(bench.constraints)
print(bench.constraint_info)
print(bench(config))
