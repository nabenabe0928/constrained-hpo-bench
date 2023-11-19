import os

from chpobench import HPOLib


print(f"{HPOLib.dataset_names=}")
print(f"{HPOLib.avail_constraint_names=}")
print(f"{HPOLib.avail_obj_names=}")
print(f"{HPOLib.config_space=}")
print(f"{HPOLib.fidel_space=}")

# You can check the constraint information by this.
print(HPOLib.get_constraint_info(HPOLib.dataset_names[0]))
bench = HPOLib(
    data_path=os.path.join(os.environ["HOME"], "hpo_benchmarks/hpolib/"),
    dataset_name=HPOLib.dataset_names[0],
    quantiles={"runtime": 0.1, "model_size": 0.5},
)

config = {
    name: config_info.choices[0]
    if hasattr(config_info, "choices")
    else config_info.seq[0]
    for name, config_info in bench.config_space.items()
}

print(bench.constraints)
print(bench(config))
