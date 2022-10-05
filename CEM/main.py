import numpy as np


def discretized_lorenz63_model(sigma, r, b, T):
    return lambda x: np.matmul(
            np.array([[1 - sigma * T, sigma * T, 0, 0, 0], [r * T, 1 - T, 0, -T, 0], [0, 0, 1 - b * T, 0, T]]), 
            np.array(([x[0], x[1], x[2], x[0] * x[2], x[0] * x[1] ]))
        )

model = discretized_lorenz63_model(10, 28, 8/3, 0.01)

N = 4000000
mu = 0
sig = np.sqrt(2)
x_t = np.zeros((N, 3))
x_t[0] = np.array([
    1.508870 + np.random.normal(mu, sig),
    -1.531271 + np.random.normal(mu, sig),
    25.46091 + np.random.normal(mu, sig)
])

# print(model(x_t[0]))

for t in range(1, N):
    x_t[t] = model(x_t[t-1])
    print(t, x_t[t])

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
