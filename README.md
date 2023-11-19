# Benchmark for Constrained Hyperparameter Optimization

This repository provides a constrained hyperparameter optimization benchmark dataset based on [HPOLib](https://github.com/automl/HPOlib1.5), [HPOBench](https://github.com/automl/hpobench), and [JAHS-Bench-201](https://github.com/automl/jahs_bench_201/).

I separately developed this benchmark because the difficulties of constrained optimization depends on [1]:
1. how tight each constraint is, and
2. how much the top-performing domain and the feasible domain overlap.

For this reason, I provide a benchmark dataset that can control the difficulties of optimization problems.

[1] [c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization](https://arxiv.org/abs/2211.14411)

## Setup

This benchmark requires Python 3.9 or later.

First, you need to install the dependencies:

```shell
$ python -m venv venv
$ pip install chpobench
```

Next, you will download the dataset (if necessary):

### HPOLib

```shell
# Install HPOLib at `YOUR_DATA_PATH`
$ cd <YOUR_DATA_PATH>
$ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
$ tar xf fcnet_tabular_benchmarks.tar.gz
$ mv fcnet_tabular_benchmarks/*.hdf5 .
$ rm -r fcnet_tabular_benchmarks/
```

Then extract the benchmark dataset in a usable format:

```python
from hpolib_extractor import extract_hpolib


# Choose epochs to use from 1 to 100.
epochs = [11, 33, 100]
data_dir = "<YOUR_DATA_PATH>"

extract_hpolib(data_dir=data_dir, epochs=epochs)
```

### HPOBench

```
$ cd <YOUR_DATA_PATH>
$ wget https://ndownloader.figshare.com/files/30379005
$ unzip nn.zip
```

Then extract the benchmark dataset in a usable format:

```python
from hpolib_extractor import extract_hpobench


# Choose epochs to use from [3, 9, 27, 81, 243].
epochs = [3, 9, 27, 81, 243]
data_dir = "<YOUR_DATA_PATH>"

extract_hpobench(data_dir=data_dir, epochs=epochs)
```


### JAHS-Bench-201

```shell
$ cd <YOUR_DATA_PATH>
$ wget https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar
# Uncompress assembled_surrogates.tar!!
```

## Benchmark Usage

Here is an example using HPOBench:

```python
import os

from chpobench import HPOBench


# Show all the available dataset names.
print(f"{HPOBench.dataset_names=}")
# Show all the available constraint names.
print(f"{HPOBench.avail_constraint_names=}")
# Show all the available objective metric names.
print(f"{HPOBench.avail_obj_names=}")
# Show the search space.
print(f"{HPOBench.config_space=}")
# Show the fidelity space.
print(f"{HPOBench.fidel_space=}")

# Check the constraint information by this.
print(HPOBench.get_constraint_info(HPOBench.dataset_names[0]))
bench = HPOBench(
    # You need to specify where you store the benchmark data downloaded above.
    data_path=os.path.join(os.environ["HOME"], "hpo_benchmarks/hpobench/"),
    # You need to give the dataset name to test.
    dataset_name=HPOBench.dataset_names[0],
    # Quantiles control the tightness of each constraint. HPOBench.avail_quantiles shows the available quantiles.
    quantiles={"runtime": 0.1, "precision": 0.5},
    # metric_names=[...]  # Less metric specification can reduce memory consumption for JAHS-Bench-201.
)

config = {name: config_info.seq[0] for name, config_info in bench.config_space.items()}

print(bench.constraints)
print(bench(config))

```

For more details, please check [the examples](examples/).
