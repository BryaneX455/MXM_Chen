from CausalityBasedSystemLearner import *

def L63_model(sigma, r, b, dt, noise):
    return lambda z, sigma=sigma, r=r, b=b, dt=dt, noise=noise: np.array([
        sigma*z[1]*dt  - sigma*z[0]*dt + z[0],
        r*z[0]*dt  - z[0]*z[2]*dt  - z[1]*dt + z[1],
        z[0]*z[1]*dt - b*z[2]*dt + z[2],
    ]) + np.random.normal(scale=noise, size=3)

def main():
    count = 250_000

    # L63 Parameters
    sigma: float = 10
    r: float = 28
    b: float = 8/3
    dt: float = 0.001
    noise = 0.001

    model = L63_model(sigma, r, b, dt, noise)

    z_0 = np.array([
        1.508870,
        -1.531271,
        25.46091,
    ]) + np.random.normal(scale=noise, size=3)

    z = np.zeros((count + 1, len(z_0)))
    z[0] = z_0
    for t in range(0, count):
        z[t+1] = model(z[t])
    
    print(z)
    
    (lib, names, H) = construct_function_library(3, polynomial_power=2)
    (Z, F) = generate_function_library_timeseries(
        z,
        lib
    )
    print(names)
    print(compute_causation_entropy_matrix(Z,F))
    xi = identify_nonzero_causation_e`ntropy_entries(Z, F, permutations=75, tqdm=lambda iter: tqdm(iter, desc="Computing permuted causation entropy"))
    print(xi)
    params = extract_parameters(xi)
    print(estimate_parameters(Z,F,params))


if __name__ == "__main__":
    main()