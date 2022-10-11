#!/usr/bin/env python

import numpy as np
from itertools import product

def main():
    sigma: float = 10
    r: float = 28
    b: float = 8/3
    dt: float = 0.01

    mu = 0
    sig = np.sqrt(2)

    L63_constants = np.array([
                [1 - sigma * dt, sigma * dt, 0, 0, 0], 
                [r * dt, 1 - dt, 0, -dt, 0], 
                [0, 0, 1 - b * dt, 0, dt]
            ])
    
    L63_function_library = [
        lambda z, t: z[0],
        lambda z, t: z[1],
        lambda z, t: z[2],
        lambda z, t: z[0] * z[2],
        lambda z, t: z[0] * z[1]
    ]

    z_0 = np.array([
        1.508870,
        -1.531271,
        25.46091
    ])

    # z_0 = np.array([
    #     1.508870 + np.random.normal(mu, sig),
    #     -1.531271 + np.random.normal(mu, sig),
    #     25.46091 + np.random.normal(mu, sig)
    # ])

    count = 500
    z = np.zeros((count + 1, len(z_0)))
    f = np.zeros((count, len(L63_function_library)))

    z[0] = z_0
    for t in range(0, count):
        f[t] = np.array(list(map(lambda x: x(z[t], t), L63_function_library)))
        z[t+1] = np.matmul(
            L63_constants,
            f[t]
        )
    z = z[:-1]
    
    M = len(L63_function_library)
    N = len(z_0)
    CEM = np.zeros((M, N))
    for (m, n) in product(range(0, M), range(0, N)):
        Z = f[:, m:m+1]
        X = z[:, n:n+1]
        rng = list(range(0, len(L63_function_library)))
        rng.remove(m)
        Y = f[:, rng]

        R_Y = np.linalg.det(np.cov(f[:, 1:5]))
        R_YZ = np.linalg.det(np.cov(np.concatenate((Y, Z), axis=1)))
        R_XY = np.linalg.det(np.cov(np.concatenate((X, Y), axis=1)))
        R_XYZ = np.linalg.det(np.cov(np.concatenate((X, Y, Z), axis=1)))
        a = [R_Y, R_YZ, R_XY, R_XYZ]
        print(m,n,a)
        if any(map(lambda x: x <= 0, a)): 
            CEM[m,n] = 0
            continue
        else:
            CEM[m,n] = 0.5 * np.log(R_XY) - 0.5 * np.log(R_Y) - 0.5 * np.log(R_XYZ) + 0.5 * np.log(R_YZ)
        
    print(CEM)

    # Z = [f0]
    # X = [z0]
    # Y = [f1,...,f4]
    # R_Y = np.linalg.det(np.cov(f[:, 1:5]))
    # R_YZ = np.linalg.det(np.cov(np.concatenate((f[:, 1:5],  f[:, 0:1]), axis=1)))
    # R_XY = np.linalg.det(np.cov(np.concatenate((z[:, 0:1], f[:, 1:5]), axis=1)))
    # R_XYZ = np.linalg.det(np.cov(np.concatenate((z[:, 0:1], f[:, 1:5], f[:, 0:1]), axis=1)))
    
    # print(R_Y, R_YZ, R_XY, R_XYZ)

    # if R_Y < 0: R_Y = 0
    # if R_YZ < 0: R_YZ = 0
    # if R_XY < 0: R_XY = 0
    # if R_XYZ < 0: R_XYZ = 0
    
    # print(R_Y, R_YZ, R_XY, R_XYZ)

    # C_ZX_Y = 0.5 * np.log(R_XY) - 0.5 * np.log(R_Y) - 0.5 * np.log(R_XYZ) + 0.5 * np.log(R_YZ)
    # print(C_ZX_Y)
    # # print(np.linalg.det(np.cov(f[:4, 1:5])))

    # model = discretized_lorenz63_model(10, 28, 8/3, 0.01)

    # count = 100
    # mu = 0
    # sig = np.sqrt(2)

    # z_0 = np.array([
    #     1.508870,
    #     -1.531271,
    #     25.46091
    # ])
    # z = np.zeros((count, 3))
    # z[0] = z_0
    # for t in range(1, count):
    #     z[t] = model(z[t-1])
    #     print(t, z[t])

    # z_0 = np.array([
    #     1.508870 + np.random.normal(mu, sig),
    #     -1.531271 + np.random.normal(mu, sig),
    #     25.46091 + np.random.normal(mu, sig)
    # ])

    # lib = library_functions(3)
    # xi = random_xi(3, len(lib))
    # print("TEST: ", np.matmul(xi, compute_library(lib)(z_0, 1)))
    # sim = simulate_model(
    #     xi,
    #     lib,
    #     z_0,
    #     count
    # )
    # print(sim)

    # print(model(x_t[0]))

    # for t in range(1, N):
    #     x_t[t] = model(x_t[t-1])
    # # print(t, x_t[t])

    # print(x_t)
    # print(x_t.transpose()[0])
    # print(gaussian_estimate(x_t.transpose()[0], x_t.transpose()[1], x_t.transpose()[2]))

if __name__ == "__main__":
    main()

# Elinger 2020 Eq. 2.6
def h(d: int, N: int):
    return ((4 / (d+2)) ** (1 / (d+4))) * N ** (-1 / (d+4))

# Elinger 2020 Eq. 2.5
def K(h: float, d: int, detS: float, x):
    return (1 / (
        (2 * np.pi)^(float(d)/2)
        * (h ** d)
        * (detS ** 0.5)
    )) * np.exp(- (x / 2))


# Chen & Zhang 2022 Eq. 6 
def discretized_lorenz63_model(sigma: float, r: float, b: float, dt: float):
    return lambda z: np.matmul(
            # Xi
            np.array([
                [1 - sigma * dt, sigma * dt, 0, 0, 0], 
                [r * dt, 1 - dt, 0, -dt, 0], 
                [0, 0, 1 - b * dt, 0, dt]
            ]),
            # F(z(t), t)
            np.array([
                z[0],
                z[1],
                z[2],
                z[0] * z[2],
                z[0] * z[1]
            ])
        )


def library_functions(N: int):
    """N from Chen & Zhang Eq. 6, the number of state variables in the vector Z.
    Outputs a list of lambdas which take a state vector z and time step t, and produce f_n(z, t).
    """
    rv = []
    
    # Linear functions
    for i in range(0, N):
        def f(z, t, i=i): return z[i]
        rv.append(f)
    
    # Quadratic functions
    for (i,j) in product(range(0, N), repeat=2):
        def f(z, t, i=i, j=j): return z[i] * z[j]
        rv.append(f)
    
    return rv

def compute_library(lib: list):
    """Converts a list of library function lambdas from i.e. library_functions into a single function of z and t"""
    def f(z,t, lib=lib): return np.array(list(map(lambda f: f(z,t), lib)))
    return f

def random_xi(N, M):
    """"""
    return 0.01 * np.random.rand(N,M)

def model_timestep(Xi, F, z_t, t: float):
    # print("Xi: ", Xi, "F: ", F(z_t, t))
    return np.matmul(Xi, F(z_t, t))

def simulate_model(Xi, lib, z_0, count: int, dt: float = 0.01):
    F = compute_library(lib)
    z = np.zeros((count, len(z_0)))
    z[0] = z_0
    for i in range(1, count):
        z[i] = model_timestep(Xi, F, z[i-1], i*dt)
        print(z[i])
    return z

# Chen and Zhang 2022 Prop. 4
# C_{Z \to X | Y}
def gaussian_estimate(z, x, y):
    R_XY = np.log(np.linalg.det(np.cov(x, y)))
    # R_Y = np.log(np.linalg.det(np.cov(y, y)))
    R_XYZ = np.log(np.linalg.det(np.cov(np.array([x,y,z]))))
    R_YZ = np.log(np.linalg.det(np.cov(y, z)))

    return 0.5 * R_XY - 0.5 * R_XYZ + 0.5 * R_YZ  # - 0.5 * R_Y