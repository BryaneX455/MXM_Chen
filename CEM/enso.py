#!/usr/bin/env python

import numpy as np
from CausalityBasedSystemLearner import *
from itertools import product
from tqdm import tqdm

def main():
    data = np.genfromtxt('ENSO_TimeSeries.csv', delimiter=',', skip_header=1)
    # (lib, names, quadratics) = construct_function_library(2, polynomial_power=2)
    lib = [
        lambda z, t: z[0],
        lambda z, t: z[1],
        lambda z, t: z[0] * z[0],
        lambda z, t: z[0] * z[1],
        lambda z, t: z[1] * z[1],
    ]

    paired_functions = [
        (3, 2), (4, 3)
    ]

    (Z, F) = generate_function_library_timeseries(
        data[:, [2,3]],
        lib
    )

    print("Causation Entropy Matrix:")
    print(compute_causation_entropy_matrix(Z,F))
    print("--------------------------------------------------")
    xi = identify_nonzero_causation_entropy_entries(
        Z, 
        F, 
        permutations=250, 
        significance_level=0.99, 
        tqdm=lambda iter: tqdm(iter, desc="Computing permuted causation entropy"))
    # print(names)
    print("Xi:")
    print(xi)
    for (fx,fy) in paired_functions:
        if xi[0][fx]: xi[1][fy] = 1
        if xi[1][fy]: xi[0][fx] = 1
    print(xi)
    print("--------------------------------------------------")
    params = extract_parameters(xi)
    print("Without Physics Constraints: ", estimate_parameters(Z,F,params))
    (lmbda, results) = estimate_parameters_with_physics_constraints(Z,F,params,paired_functions)
    print("With Physics Constraints: λ = ", lmbda, " ", results)
    # print(list(map(lambda x: reduce(lambda a,b: a+" + "+b, x), format_equations(params, names, results))))

if __name__ == "__main__":
    main()