import numpy as np
from tqdm import tqdm
from itertools import product, repeat, combinations_with_replacement
from functools import reduce
from multiprocessing import Pool

def construct_function_library(
    N,
    polynomial_power=2,
    # trig_functions=False,
):
    rv = []
    for k in range(1, polynomial_power + 1):
        for multi_index in combinations_with_replacement(range(0,N), k):
            name = reduce(lambda a,b: a+"*"+b, map(lambda x: f'z[{x}]', multi_index))
            def f(z,t, multi_index=multi_index):
                return reduce(lambda a,b: a*b, map(lambda j: z[j], multi_index))
            rv.append((f, name, k == 2))

    return (
        list(map(lambda x: x[0], rv)),
        list(map(lambda x: x[1], rv)),
        list(map(lambda x: x[2], rv)),
    )

def generate_function_library_timeseries(
    z,
    function_library,
):
    J = len(z)
    f = np.zeros((J, len(function_library)))
    for j in range(0, J):
        f[j] = np.array(list(map(lambda x: x(z[j-1], j-1), function_library)))
    return (z[1:], f[1:])

# Gaussian estimate of the causation entropy from Z to X conditioned on Y.
def causation_entropy_estimate_gaussian(Z, X, Y):
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

def compute_causation_entropy_matrix(
    z,
    f,
    causation_entropy_estimator=causation_entropy_estimate_gaussian,
    tqdm=lambda iter: iter,
):
    # Number of functions in the function library
    M = len(f[0])
    # Number of state variables
    N = len(z[0])
    
    rv = np.zeros((N, M))
    for (n, m) in tqdm(product(range(0, N), range(0, M))):
        # F\{f_m}
        indices = list(range(0, len(f[0])))
        indices.remove(m)

        # F_m to Z_n | F\{F_m}
        rv[n,m] = causation_entropy_estimator(
            f[:, m:m+1],
            z[:, n:n+1],
            f[:, indices]
        )

    return rv

"""Permute a time series along the time axis (first coordinate)"""
def permute_time_series(x, rng):
    return np.concatenate(
            list(map(lambda i: rng.permutation(x[:, [i]]), 
                range(0, len(x[0]))
            )),
            axis=1
        )
        

"""Takes (Z, F, causation_entropy_estimator, rng)"""
def __permuted_CEM_helper(args):
    return compute_causation_entropy_matrix(
        permute_time_series(args[0], args[3]),
        permute_time_series(args[1], args[3]),
        args[2]
    )

"""Generate `self.permutations` many random permutations of the timeseries `self.Z` and `self.F`, and then compute the causation entropy matrices on these permuted time series and store them in `self.permuted_causation_entropies`
"""
def compute_permuted_causation_entropies(
    z,
    f,
    causation_entropy_estimator=causation_entropy_estimate_gaussian,
    permutations = 100,
    rng = lambda s: np.random.default_rng(s),
    processes = 8,
    tqdm = lambda iter: iter,
):
    with Pool(processes) as p:
        return np.array(list(
            tqdm(
                p.imap_unordered(
                    __permuted_CEM_helper,
                    map(
                        lambda i: (z, f, causation_entropy_estimator, rng(i+3)),
                        range(0, permutations)
                    )
                )
            )
        ))

"""Permutation test.

Estimate the causation entropy matrix. Then count the number of permutations which are less than estimated causation entropy. If the proportion of permutations which are less than the estimated causation entropy is greater than the significance level, then decide that this entry has strictly positive causation entropy. Otherwise, decide it has zero causation entropy.
"""
def identify_nonzero_causation_entropy_entries(
    z,
    f,
    causation_entropy_estimator=causation_entropy_estimate_gaussian,
    permutations = 100,
    significance_level = 0.99,
    rng = lambda s: np.random.default_rng(s),
    processes = 8,
    tqdm = lambda iter: iter,
):
    CEM = compute_causation_entropy_matrix(z, f)
    permuted_causation_entropies = compute_permuted_causation_entropies(
        z,
        f,
        causation_entropy_estimator,
        permutations,
        rng,
        processes,
        tqdm)

    Xi = np.zeros(CEM.shape)
    for (m,n) in product(range(0, CEM.shape[0]), range(0, CEM.shape[1])):
        # See: Sun et. al. 2014 p. 3423
        count: float = len(list(
            filter(lambda x: x <= CEM[m][n],
            # Project to single entry of permuted_causation_entropies
            map(lambda x: x[m][n], permuted_causation_entropies))
        ))
        prop = count / float(permutations)
        Xi[m][n] = prop > significance_level

    return Xi

def extract_parameters(
    xi
):
    iter = np.nditer(xi, order='C', flags=['multi_index'])
     # Filter for nonzero entries in xi and then
     # just return the multiindex.
    return list(map(lambda x: x[1],
        filter(lambda x: x[0] != 0,
        map(lambda x: (x, iter.multi_index), iter))))

"""M from Chen & Zhang 2022 Eq. 12,14"""
def construct_parameter_estimation_matrices(
    z,
    f,
    parameter_indices,
):
    M = np.zeros((len(z), len(parameter_indices), len(z[0])))
    for j in range(0, len(z)):
        for i, t in enumerate(parameter_indices):
            M[j][i][t[0]] = f[j][t[1]]
    return M

def estimate_sigma(
    z,
):
    return reduce(
            lambda a, b: a+b,
            map(
                lambda j: np.matmul((z[j+1] - z[j]).transpose(), (z[j+1] - z[j])), 
                range(0, len(z) - 1)
            )
        ) / len(z)

def estimate_parameters(
    z,
    f,
    parameter_indices
):
    Sigma = estimate_sigma(z)
    M = construct_parameter_estimation_matrices(z,f,parameter_indices)
    D = (1/Sigma) * reduce(
        lambda a,b: a+b, 
        map(
            lambda j: np.matmul(M[j], M[j].transpose()), 
            range(0, len(z))
        )
    )
    c = (1/Sigma) * reduce(
        lambda a,b: a+b, 
        map(
            lambda j: np.matmul(M[j], (z[j+1] - z[j])), 
            range(0, len(z) - 1)
        )
    )

    return np.matmul(np.linalg.inv(D), c)

def estimate_parameters_with_physics_constraints(
    z,
    f,
    parameter_indices,
    quadratics,
):
    Sigma = estimate_sigma(z)
    M = construct_parameter_estimation_matrices(z,f,parameter_indices)
    D = (1/Sigma) * reduce(
        lambda a,b: a+b, 
        map(
            lambda j: np.matmul(M[j], M[j].transpose()), 
            range(0, len(z))
        )
    )
    c = (1/Sigma) * reduce(
        lambda a,b: a+b, 
        map(
            lambda j: np.matmul(M[j], (z[j+1] - z[j])), 
            range(0, len(z) - 1)
        )
    )
    H = np.array(list(map(lambda x: 1 if quadratics[x[1]] else 0, parameter_indices)))

    lmbda = np.matmul(
        np.matmul(H.transpose(),
        np.linalg.inv(D)
        ), H) / np.matmul(np.matmul(H, np.linalg.inv(D)), c)
    print(lmbda)
    return np.matmul(
            np.linalg.inv(D),
            (c - lmbda * H.transpose())
        )

class CausalityBasedSystemLearner:
    """Initialize a CausalityBasedSystemLearner

    Z is the observed values of the state variables. First index is t, second index is the number of state variables.
    function_library is a list of functions that depends on (z[t],t).
    function_library_quadratics is a list of booleans used to identify which parameters are held consant in the physics constraints.
    """
    def __init__(
        self,
        Z,
        function_library,
        function_library_quadratics = None,
        causation_entropy_estimator = causation_entropy_estimate_gaussian,
        permutations=100,
        significance_level=0.99,
        processes = 8,
    ):
        # shape: (timesteps, variables)
        self.Z = Z
        self.function_library = function_library
        self.function_library_quadratics = function_library_quadratics
        self.causation_entropy_estimator = causation_entropy_estimator
        self.permutations = permutations
        self.significance_level = significance_level
        self.processes = processes
        
        if self.function_library_quadratics != None:
            assert len(self.function_library) == len(self.function_library_quadratics)

        # Number of observations
        J = len(self.Z)
        self.F = np.zeros((J, len(self.function_library)))
        for j in tqdm(range(0, J)):
            self.F[j] = np.array(list(map(lambda x: x(self.Z[j-1], j-1), self.function_library)))
        
        # Save Z_0 and discard it from the data
        # self.Z_0 = Z[0]
        # self.F = self.F[1:]
        # self.Z = self.Z[1:]

    ###
    # Causation Entropy
    ###

    # def compute_causation_entropy_matrix(
    #     self,
    #     tqdm=lambda iter: tqdm(iter, desc="Computing causation entropy matrix"),
    # ):
    #     return compute_causation_entropy_matrix(
    #         self.Z,
    #         self.F,
    #         tqdm)

    ###
    # Permutation Test
    ###

    def estimate_parameters(self):
        Xi = self.identify_nonzero_causation_entropy_entries()
        # CEM = self.compute_causation_entropy_matrix()

        it = np.nditer(CEM_b, order='C', flags=['multi_index'])
        Theta = list(map(lambda x: x[1], filter(lambda x: x[0] != 0, map(lambda x: (x, it.multi_index), it))))
        H = np.array(list(map(lambda x: 1 if self.function_library_quadratics[x[1]] else 0, Theta)))

        # Chen and Zhang (2022) Eqns. 15 and 16
        M = np.zeros((len(self.Z), len(Theta), len(self.Z[0])))
        for j in range(0, len(self.Z)):
            for i, t in enumerate(Theta):
                M[j][i][t[0]] = self.F[j][t[1]]

        Sigma = reduce(
            lambda a, b: a+b, 
            map(
                lambda j: np.matmul((self.Z[j+1] - self.Z[j]).transpose(), (self.Z[j+1] - self.Z[j])), 
                range(0, len(self.Z) - 1)
            )
        ) / len(self.Z)

        D = reduce(lambda a,b: a+b, map(lambda j: (1/Sigma) * np.matmul(M[j], M[j].transpose()), range(0, len(self.Z))))
        c = reduce(lambda a,b: a+b, map(lambda j: (1/Sigma) * np.matmul(M[j], (self.Z[j+1] - self.Z[j])), range(0, len(self.Z) - 1)))

        lmbda = np.matmul(np.matmul(H.transpose(), np.linalg.inv(D)), H) / np.matmul(np.matmul(H, np.linalg.inv(D)), c)

        result = np.zeros(CEM.shape)
        for theta, loc in zip(np.matmul(
            np.linalg.inv(D),
            (c - lmbda * H.transpose())
        ), Theta):
            result[loc] = theta
        
        return result
    
    

        
