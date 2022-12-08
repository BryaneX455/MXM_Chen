#!/usr/bin/env python

import numpy as np
from CausalityBasedSystemLearner import *
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io

def main():
    # mat = scipy.io.loadmat('Obs_ENSO.mat')
    # print(mat.keys())
    # print(mat['h_W_new'])

    # exit(0)

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

    normalized_data = data[:, [2,3]] / np.std(data[:, [2,3]], axis=0)
    (Z, F) = generate_function_library_timeseries(
        normalized_data,
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

    # Simulate futures and generate histograms
    future = simulate_future(Z[0], lib, theta/365, sigma, count=normalized_data.shape[0])
    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    axs[0][0].hist(normalized_data[:, 0], bins=100)
    axs[0][0].title.set_text("Therm. Depth (Actual, Normalized)")
    axs[0][1].hist(normalized_data[:, 1], bins=100)
    axs[0][1].title.set_text("S.s. Temp. (Actual, Normalized)")
    axs[1][0].hist(future[:, 0], bins=100)
    axs[1][0].title.set_text("Thermocline Depth (Simulated)")
    axs[1][1].hist(future[:, 1], bins=100)
    axs[1][1].title.set_text("Seasurface Temperature (Simulated)")
    plt.savefig("hist.png")

    # Line plot
    plt.clf()
    fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)
    x = range(0, normalized_data.shape[0])
    axs[0].plot(x, normalized_data[:, 0], label="Actual Thermocline Depth")
    axs[0].plot(x, future[:, 0], label="Simulated Thermocline Depth")
    axs[0].legend()

    axs[1].plot(x, normalized_data[:, 1], label="Actual Seasurface Temperature")
    axs[1].plot(x, future[:, 1], label="Simulated Seasurface Temperature")
    axs[1].legend()
    plt.savefig("line.png")

def simulate_future(
    z_0,
    function_library,
    theta,
    sigma,
    count=1000,
):
    dt = 1/365
    z = z_0
    rv = np.zeros((count, len(z_0)))
    for t in range(0, count):
        f = np.array(list(map(lambda l: l(z, t), function_library)))
        z = dt * np.matmul(theta, f) + z + np.sqrt(dt) * np.matmul(sigma, np.random.normal(size=2))
        rv[t] = z
    return rv

if __name__ == "__main__":
    main()