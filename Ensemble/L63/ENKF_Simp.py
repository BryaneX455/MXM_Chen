# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 11:52:34 2022

@author: Bryan
"""

import numpy as np

def EnKF(ubi,w,ObsOp,JObsOp,R,B):
    # The analysis step for the (stochastic) ensemble Kalman filter
    # with virtual observations
    n,N = ubi.shape # n is the state dimension and N is the size of ensemble
    m = w.shape[0] # m is the size of measurement vector
    # compute the mean of forecast ensemble
    ub = np.mean(ubi,1)
    # compute Jacobian of observation operator at ub
    Dh = JObsOp(ub)
    # compute Kalman gain
    D = Dh@B@Dh.T + R
    K = B @ Dh @ np.linalg.inv(D)
    wi = np.zeros([m,N])
    uai = np.zeros([n,N])
    for i in range(N):
        # create virtual observations
        wi[:,i] = w + np.random.multivariate_normal(np.zeros(m), R)
        # compute analysis ensemble
        uai[:,i] = ubi[:,i] + K @ (wi[:,i]-ObsOp(ubi[:,i]))
    # compute the mean of analysis ensemble
    ua = np.mean(uai,1)
    # compute analysis error covariance matrix
    P = (1/(N-1)) * (uai - ua.reshape(-1,1)) @ (uai - ua.reshape(-1,1)).T
    return uai, P
