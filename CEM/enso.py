#!/usr/bin/env python

import numpy as np
from CausalityBasedSystemLearner import *
from itertools import product
from tqdm import tqdm

def main():
    data = np.genfromtxt('ENSO_TimeSeries.csv', delimiter=',', skip_header=1)
    (lib, names, quadratics) = construct_function_library(2, polynomial_power=3)
    (Z, F) = generate_function_library_timeseries(
        data[:, [2,3]],
        lib
    )

    print("Causation Entropy Matrix:")
    print(compute_causation_entropy_matrix(Z,F))
    print("--------------------------------------------------")
    xi = identify_nonzero_causation_entropy_entries(Z, F, permutations=250, significance_level=0.99, tqdm=lambda iter: tqdm(iter, desc="Computing permuted causation entropy"))
    print(names)
    print("Xi:")
    print(xi)
    print("--------------------------------------------------")
    params = extract_parameters(xi)
    print("Without Physics Constraints: ", estimate_parameters(Z,F,params))
    (lmbda, results) = estimate_parameters_with_physics_constraints(Z,F,params,quadratics)
    print("With Physics Constraints: λ = ", lmbda, " ", results)

if __name__ == "__main__":
    main()