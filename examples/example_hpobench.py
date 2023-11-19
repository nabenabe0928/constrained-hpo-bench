import os

from chpobench import HPOBench


bench = HPOBench(
    data_path=os.path.join(os.environ["HOME"], "hpo_benchmarks/hpobench/"),
    dataset_name="australian",
    quantiles={"runtime": 0.1, "precision": 0.5},
    metric_names=["runtime", "loss", "precision"],
)

config = {name: config_info.seq[0] for name, config_info in bench.config_space.items()}

print(bench.constraints)
print(bench.constraint_info)
print(bench(config))
