import os

from chpobench import JAHSBench201


print(f"{JAHSBench201.dataset_names=}")
print(f"{JAHSBench201.avail_constraint_names=}")
print(f"{JAHSBench201.avail_obj_names=}")
print(f"{JAHSBench201.directions=}")
print(f"{JAHSBench201.config_space=}")
print(f"{JAHSBench201.fidel_space=}")

# Check the constraint information by this.
print(JAHSBench201.get_constraint_info(JAHSBench201.dataset_names[0]))
bench = JAHSBench201(
    data_path=os.path.join(os.environ["HOME"], "hpo_benchmarks/jahs/"),
    dataset_name=JAHSBench201.dataset_names[0],
    quantiles={"runtime": 0.1, "model_size": 0.5},
    # metric_names=[...]  # less metric usage reduces the memory consumption and runtime for JAHS-Bench-201.
)

config = {
    name: config_info.choices[0]
    if hasattr(config_info, "choices")
    else (config_info.seq[0] if hasattr(config_info, "seq") else config_info.lower)
    for name, config_info in bench.config_space.items()
}

print(bench.constraints)
print(bench(config))
