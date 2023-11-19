import os

from chpobench import JAHSBench201


bench = JAHSBench201(
    data_path=os.path.join(os.environ["HOME"], "hpo_benchmarks/jahs/"),
    dataset_name="cifar10",
    quantiles={"runtime": 0.1, "model_size": 0.5},
    metric_names=["runtime", "loss", "model_size"],
)

config = {
    "LearningRate": 1e-3,
    "WeightDecay": 1e-5,
    "N": 1,
    "W": 4,
    "Activation": "ReLU",
    "TrivialAugment": True,
    "Op1": 0,
    "Op2": 0,
    "Op3": 0,
    "Op4": 0,
    "Op5": 0,
    "Op6": 0,
}

print(bench.constraints)
print(bench(config))
