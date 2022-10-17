#!/usr/bin/env python

import numpy as np
from itertools import product

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

def main():
    sigma: float = 10
    r: float = 28
    b: float = 8/3
    dt: float = 0.001

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

    count = 100_000
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

    print("CEM: ", CEM.transpose())
    # print("Chosen Parameters: ", CEM_b.transpose())
    print("Original Constants: ", L63_constants)

if __name__ == "__main__":
    main()

# # Elinger 2020 Eq. 2.6
# def h(d: int, N: int):
#     return ((4 / (d+2)) ** (1 / (d+4))) * N ** (-1 / (d+4))

# # Elinger 2020 Eq. 2.5
# def K(h: float, d: int, detS: float, x):
#     return (1 / (
#         (2 * np.pi)^(float(d)/2)
#         * (h ** d)
#         * (detS ** 0.5)
#     )) * np.exp(- (x / 2))


# # Chen & Zhang 2022 Eq. 6 
# def discretized_lorenz63_model(sigma: float, r: float, b: float, dt: float):
#     return lambda z: np.matmul(
#             # Xi
#             np.array([
#                 [1 - sigma * dt, sigma * dt, 0, 0, 0], 
#                 [r * dt, 1 - dt, 0, -dt, 0], 
#                 [0, 0, 1 - b * dt, 0, dt]
#             ]),
#             # F(z(t), t)
#             np.array([
#                 z[0],
#                 z[1],
#                 z[2],
#                 z[0] * z[2],
#                 z[0] * z[1]
#             ])
#         )


# def library_functions(N: int):
#     """N from Chen & Zhang Eq. 6, the number of state variables in the vector Z.
#     Outputs a list of lambdas which take a state vector z and time step t, and produce f_n(z, t).
#     """
#     rv = []
    
#     # Linear functions
#     for i in range(0, N):
#         def f(z, t, i=i): return z[i]
#         rv.append(f)
    
#     # Quadratic functions
#     for (i,j) in product(range(0, N), repeat=2):
#         def f(z, t, i=i, j=j): return z[i] * z[j]
#         rv.append(f)
    
#     return rv

# def compute_library(lib: list):
#     """Converts a list of library function lambdas from i.e. library_functions into a single function of z and t"""
#     def f(z,t, lib=lib): return np.array(list(map(lambda f: f(z,t), lib)))
#     return f

# def random_xi(N, M):
#     """"""
#     return 0.01 * np.random.rand(N,M)

# def model_timestep(Xi, F, z_t, t: float):
#     # print("Xi: ", Xi, "F: ", F(z_t, t))
#     return np.matmul(Xi, F(z_t, t))

# def simulate_model(Xi, lib, z_0, count: int, dt: float = 0.01):
#     F = compute_library(lib)
#     z = np.zeros((count, len(z_0)))
#     z[0] = z_0
#     for i in range(1, count):
#         z[i] = model_timestep(Xi, F, z[i-1], i*dt)
#         print(z[i])
#     return z

# # Chen and Zhang 2022 Prop. 4
# # C_{Z \to X | Y}
# def gaussian_estimate(z, x, y):
#     R_XY = np.log(np.linalg.det(np.cov(x, y)))
#     # R_Y = np.log(np.linalg.det(np.cov(y, y)))
#     R_XYZ = np.log(np.linalg.det(np.cov(np.array([x,y,z]))))
#     R_YZ = np.log(np.linalg.det(np.cov(y, z)))

#     return 0.5 * R_XY - 0.5 * R_XYZ + 0.5 * R_YZ  # - 0.5 * R_Y