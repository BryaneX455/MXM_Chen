#!/usr/bin/env python

import numpy as np
from CausalityBasedSystemLearner import *
from itertools import product
from tqdm import tqdm

def main():
    data = np.genfromtxt('ENSO_TimeSeries.csv', delimiter=',', skip_header=1)
    # (lib, names, quadratics) = construct_function_library(2, polynomial_power=2)
    lib = [
        lambda z, t: z[0],          # x
        lambda z, t: z[1],          # y
        lambda z, t: z[0] * z[0],   # x^2
        lambda z, t: z[0] * z[1],   # xy
        lambda z, t: z[1] * z[1],   # y^2
    ]

    paired_functions = [
        (3, 2), (4, 3)
    ]

    (Z, F) = generate_function_library_timeseries(
        data[:, [2,3]] / np.std(data[:, [2,3]], axis=0), # Normalized data
        lib
    )

    # print("Causation Entropy Matrix:")
    # print(compute_causation_entropy_matrix(Z,F))
    # print("--------------------------------------------------")
    xi = identify_nonzero_causation_entropy_entries(
        Z,
        F,
        permutations=250,
        significance_level=0.99,
        tqdm=lambda iter: tqdm(iter, desc="Computing permuted causation entropy"))
    # print(names)
    # print("Xi:")
    # print(xi)
    for (fx,fy) in paired_functions:
        if xi[0][fx]: xi[1][fy] = 1
        if xi[1][fy]: xi[0][fx] = 1
    # print(xi)
    print("--------------------------------------------------")
    params = extract_parameters(xi)
    (sigma, results) = estimate_parameters(Z,F,params,paired_functions,physics_constraints=True)
    # print(params)
    theta = np.zeros((2, len(lib)))
    for (i, (j, k)) in enumerate(params):
        theta[j][k] = results[i]
    print("Xi: ", theta)
    print("Sigma: ", sigma)

    print(simulate_future(Z[0], lib, theta/365, sigma, count=50000))

def simulate_future(
    z_0,
    function_library,
    theta,
    sigma,
    count=1000,
):
    dt = np.sqrt(1/365)
    z = z_0
    rv = np.zeros((count, len(z_0)))
    for t in range(0, count):
        f = np.array(list(map(lambda l: l(z, t), function_library)))
        z = np.matmul(theta, f) + z + dt * np.matmul(sigma, np.random.normal(size=2))
        rv[t] = z
    return rv

if __name__ == "__main__":
    main()