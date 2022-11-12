#!/usr/bin/env python

import numpy as np
from CausalityBasedSystemLearner import *
from itertools import product
from tqdm import tqdm

def main():
    data = np.genfromtxt('ENSO_TimeSeries.csv', delimiter=',', skip_header=1)
    (lib, quadratics) = construct_function_library(2, polynomial_power=3)
    (Z, F) = generate_function_library_timeseries(
        data[:, [2,3]],
        lib
    )

    print(identify_nonzero_causation_entropy_entries(Z, F, permutations=250, tqdm=lambda iter: tqdm(iter, desc="Computing permuted causation entropy")))

if __name__ == "__main__":
    main()