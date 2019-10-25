This repository contains the source codes used during the seminar on Evolutionary algorithms I.

There is a number of files:
- `sga.py` is the basic implementation of the Simple Genetic Algorithm from scratch
- `partition.py` is the implementation of an evolutionary algorithm for the set partition problem - partitioning a set of natural numbers into `K = 10` subsets with the same sum
- `cont_optim.py` is the implementation of a basic evolutionary algorithm for continuous optimization
- `co_functions.py` is the implementation of a few continuous optimization benchmarks
- `utils.py` contains implementation of simple utilities for logging the progress of the evolutionary algorithm and for making plots for comparison of multiple EAs
- `plotting.py` contains a simple script intended to be edited in order to create plots of any experiments using the stored log files