#!/usr/bin/env python

import numpy as np
from CausalityBasedSystemLearner import *
from itertools import product
from tqdm import tqdm

def main():
    data = np.genfromtxt('ENSO_TimeSeries.csv', delimiter=',', skip_header=1)
    Z = data[:, [2,3]]
    # print(Z[:, [0]])

    # z = [thermocline_depth, seasurface_temperature]
    # N = 2
    # function_library = []
    # for i in range(0, N):
    #     def f(z, t, i=i):
    #         return z[i]
    #     function_library.append(f)
    
    # for i, j in product(range(0,N), range(0,N)):
    #     def f(z, t, i=i, j=j):
    #         return z[i] * z[j]
    #     function_library.append(f)

    (Z, F) = generate_function_library_timeseries(Z, construct_function_library(2, polynomial_power=3))
    print(F.shape, permute_time_series(F, np.random.default_rng()).shape)

    # print(np.random.default_rng().permutation(Z[:, [0]]))

    # N = len(x[0])
    # return np.fromiter(
    #         map(lambda i: np.random.default_rng().permutation(x[:, [i]]), 
    #             range(0, N)
    #         ),
    #         dtype=(float, N)
    #     )

    # print(Z.shape, F.shape)
    # print(compute_causation_entropy_matrix(Z, F))
    print(identify_nonzero_causation_entropy_entries(Z, F, tqdm=lambda iter: tqdm(iter, desc="Computing permuted causation entropy")))

if __name__ == "__main__":
    main()