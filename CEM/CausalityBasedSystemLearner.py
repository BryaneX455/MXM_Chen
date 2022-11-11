import numpy as np
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool

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
    ):
        # shape: (timesteps, variables)
        self.Z = Z
        self.function_library = function_library
        self.function_library_quadratics = function_library_quadratics
        self.causation_entropy_estimator = causation_entropy_estimator
        
        if self.function_library_quadratics != None:
            assert len(self.function_library) == len(self.function_library_quadratics)

    def compute_function_values(
        self,
        tqdm=lambda iter: tqdm(iter, desc="Computing function library values"),
    ):
        # Number of observations
        J = len(self.Z)
        self.F = np.zeros((J, len(self.function_library)))
        for j in tqdm(range(1, J)):
            self.F[j] = np.array(list(map(lambda x: x(self.Z[j-1], j-1), self.function_library)))

    def __compute_causation_entropy_matrix(
        z,
        f,
        causation_entropy_estimator,
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

    def compute_causation_entropy_matrix(
        self,
        tqdm=lambda iter: tqdm(iter, desc="Computing causation entropy matrix"),
    ):
        return CausalityBasedSystemLearner.__compute_causation_entropy_matrix(self.Z, self.F, self.causation_entropy_estimator, tqdm)

    """Permute a time series along the time axis (first coordinate)"""
    def permute_time_series(x, rng):
        return np.array(
            list(
                map(lambda i: rng.permutation(x.transpose()[i]), 
                    range(0, len(x[0]))
                )
            )
        ).transpose()

    """Takes (self, tqdm)

    """
    def permuted_CEM_helper(args):
        return CausalityBasedSystemLearner.__compute_causation_entropy_matrix(
            CausalityBasedSystemLearner.permute_time_series(args[0], args[3]), 
            CausalityBasedSystemLearner.permute_time_series(args[1], args[3]),
            args[2]
        )

    # Permutation test
    def identify_nonzero_causation_entropy_entries(
        self,
        permutations,
        rng = np.random.default_rng(),
        tqdm = lambda iter: tqdm(iter, desc="Computing permuted causation entropy matrices"),
        threads = 8
    ):
        CEM_permuted = []
        with Pool(threads) as p:
            CEM_permuted = list(
                tqdm(p.imap(
                        CausalityBasedSystemLearner.permuted_CEM_helper, 
                        map(
                            lambda x: (self.Z, self.F, self.causation_entropy_estimator, rng), 
                            range(0, permutations)
                        )
                    ))
            )
        
