#!/usr/bin/env python

import numpy as np
from itertools import product
from multiprocessing import Pool

# Gaussian estimate of the causation entropy from Z to X conditioned on Y.
def gaussian_estimate(Z, X, Y):
    R_Y = np.linalg.det(np.cov(Y.transpose()))
    R_YZ = np.linalg.det(np.cov(np.concatenate((Y, Z), axis=1).transpose()))
    R_XY = np.linalg.det(np.cov(np.concatenate((X, Y), axis=1).transpose()))
    R_XYZ = np.linalg.det(np.cov(np.concatenate((X, Y, Z), axis=1).transpose()))

    # Adjust for numerical issues. A negative determinant from a 
    # covariance matrix can only be a result of a floating point rounding error.
    # So just set the covariance to a very small positive epsilon.
    if R_Y <= 0: R_Y = np.finfo(float).eps
    if R_YZ <= 0: R_YZ = np.finfo(float).eps
    if R_XY <= 0: R_XY = np.finfo(float).eps
    if R_XYZ <= 0: R_XYZ = np.finfo(float).eps

    return 0.5 * np.log(R_XY) - 0.5 * np.log(R_Y) - 0.5 * np.log(R_XYZ) + 0.5 * np.log(R_YZ)

def calculate_CEM(z, f, cem_estimator):
    M = len(f[0])
    N = len(z[0])
    rv = np.zeros((M, N))
    for (m, n) in product(range(0, M), range(0, N)):
        Z = f[:, m:m+1]
        X = z[:, n:n+1]
        rng = list(range(0, len(f[0])))
        rng.remove(m)
        Y = f[:, rng]

        rv[m,n] = cem_estimator(Z, X, Y)

    return rv

# Permute a np array where the first coordinate is the time, and the second 
# coordinate is the vector index.
def permute_time_series(x):
    return np.array(
        list(
            map(lambda i: np.random.default_rng().permutation(x.transpose()[i]), 
                range(0, len(x[0]))
            )
        )
    ).transpose()

def compute_permuted_CEM(args):
    return calculate_CEM(permute_time_series(args[0]), permute_time_series(args[1]), args[2])

def main():
    count = 10_000_000

    # L63 Parameters
    sigma: float = 10
    r: float = 28
    b: float = 8/3
    dt: float = 0.001

    # Number of permutations in permutation test
    permutations = 10
    significance_level = 0.999

    mu = 0
    sig = np.sqrt(2)

    L63_constants = np.array([
                [1 - sigma * dt,    sigma * dt, 0,          0, 0,   0], 
                [r * dt,            1 - dt,     0,          0, 0,   -dt], 
                [0,                 0,          1 - b * dt, 0, dt,  0]
            ])
    
    L63_function_library = [
        lambda z, t: z[0],
        lambda z, t: z[1],
        lambda z, t: z[2],
        lambda z, t: z[0] * z[0],
        lambda z, t: z[0] * z[1],
        lambda z, t: z[0] * z[2],
    ]

    # z_0 = np.array([
    #     1.508870,
    #     -1.531271,
    #     25.46091
    # ])

    z_0 = np.array([
        1.508870 + np.random.normal(mu, sig),
        -1.531271 + np.random.normal(mu, sig),
        25.46091 + np.random.normal(mu, sig)
    ])

    z = np.zeros((count + 1, len(z_0)))
    f = np.zeros((count, len(L63_function_library)))

    z[0] = z_0
    for t in range(0, count):
        f[t] = np.array(list(map(lambda x: x(z[t], t), L63_function_library)))
        z[t+1] = np.matmul(
            L63_constants,
            f[t]
        )
    z = z[1:]
    
    CEM = calculate_CEM(z, f, gaussian_estimate)

    CEM_permuted = []
    with Pool(8) as p:
        CEM_permuted = p.map(compute_permuted_CEM, map(lambda x: (z,f,gaussian_estimate), range(0, permutations)))
        # for s in :
        #     # Permute f and z together:
        #     # d = np.random.default_rng().permutation(np.concatenate((z, f), axis=1))
        #     # z_p = d[:len(z)]
        #     # f_p = d[len(z):]
        #     print("Computing permutation ", s+1, "/", permutations)
        #     CEM_permuted.append(calculate_CEM(permute_time_series(z), permute_time_series(f), gaussian_estimate))
    
    CEM_b = np.zeros(CEM.shape)
    for (m,n) in product(range(0, CEM.shape[0]), range(0, CEM.shape[1])):
        # See: Sun et. al. 2014 p. 3423
        a: float = len(list(filter(lambda x: x <= CEM[m][n], map(lambda x: x[m][n], CEM_permuted))))
        F_C = a / float(permutations)
        # print(m,n, F_C)
        CEM_b[m][n] = F_C > significance_level

    print("CEM: ", CEM.transpose())
    print("Chosen Parameters: ", CEM_b.transpose())
    print("Original Constants: ", L63_constants)

if __name__ == "__main__":
    main()