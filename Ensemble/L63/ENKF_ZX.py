# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:15:03 2022

@author: Bryan
"""
import numpy as np

class Ensemble_Kalman_Filter():
    """
        This class implements the Ensemble Kalman Filter (ENKF). The ENKF uses certain number of state vectors (ensembles) 
        around the estimate to perform prediction and filter steps.
        
        Inputs:
            x_init: 
                (x_dim,1)
                an initial condition as starting state
            
            x_init_cov:
                ((x_dim,x_dim))
                covariance of x_init. Theoretically can be any matrix of this dimension, not very important
        ------------------------------------------------------------------------------------------------------------------
        
    """
    def __init__(self, x_init, x_init_cov, o_dim, dt, en_num, t2o, fx):
        if o_dim <= 0:
            raise ValueError('dim_z must be greater than zero')
        if en_num <= 1:
            raise ValueError('N must be greater than 1')
        if np.array(x_init).ndim != 1:
            raise ValueError('x must be a 1D array')
            
        self.x_dim = len(x_init)
        self.o_dim = len(o_dim)
        self.en_num = en_num
        self.t2o = t2o
        self.fx = fx
        