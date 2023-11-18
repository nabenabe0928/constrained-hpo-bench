# Benchmark for Constrained Hyperparameter Optimization

This repository provides a constrained hyperparameter optimization benchmark dataset based on [HPOLib](https://github.com/automl/HPOlib1.5) and [JAHS-Bench-201](https://github.com/automl/jahs_bench_201/).

I separately developed this benchmark because the difficulties of constrained optimization depends on [1]:
1. how tight each constraint is, and
2. how much the top-performing domain and the feasible domain overlap.

For this reason, I provide a benchmark dataset that can control the difficulties of optimization problems.

[1] [c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization](https://arxiv.org/abs/2211.14411)

## Setup

First, you need to install the dependencies:

```shell
$ python -m venv venv
$ pip install -r requirements.txt
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

### JAHS-Bench-201

```shell
$ cd <YOUR_DATA_PATH>
$ wget https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar
# Uncompress assembled_surrogates.tar!!
```

## Benchmark Usage

0.1, 0.5, 1, 5, 10, 25, 50, 75, 90, 95, 100
