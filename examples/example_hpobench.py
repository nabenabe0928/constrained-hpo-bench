import os

from chpobench import HPOBench


print(f"{HPOBench.dataset_names=}")
print(f"{HPOBench.avail_constraint_names=}")
print(f"{HPOBench.avail_obj_names=}")
print(f"{HPOBench.config_space=}")
print(f"{HPOBench.fidel_space=}")

# You can check the constraint information by this.
print(HPOBench.get_constraint_info(HPOBench.dataset_names[0]))
bench = HPOBench(
    data_path=os.path.join(os.environ["HOME"], "hpo_benchmarks/hpobench/"),
    dataset_name=HPOBench.dataset_names[0],
    quantiles={"runtime": 0.1, "precision": 0.5},
    metric_names=HPOBench.avail_obj_names,
)

config = {name: config_info.seq[0] for name, config_info in bench.config_space.items()}

print(bench.constraints)
print(bench(config))
