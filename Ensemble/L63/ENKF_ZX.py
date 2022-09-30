# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:15:03 2022

@author: Bryan
"""
import numpy as np
from numpy.random import multivariate_normal
from helpers import outer_product_sum
class Ensemble_Kalman_Filter():
    """
        This class implements the Ensemble Kalman Filter (ENKF). The ENKF uses certain number of 
        state vectors (ensembles) around the estimate to perform prediction and filter steps.
        
        Inputs:
            x_init: 
                (x_dim,1)
                an initial condition as starting state
            
            x_init_cov:
                ((x_dim,x_dim))
                covariance of x_init. 
                Theoretically can be any matrix of this dimension, not very important
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
        self.dt = dt
        self.G = np.zeros((self.x_dim, self.o_dim))
        self.xo = np.array([[None] * self.o_dim]).T
        self.x = x_init
        self.C = x_init_cov
        
        self.x_prior = self.x.copy()
        self.C_prior = self.C.copy()
        
        self.x_post = self.x.copy()
        self.C_post = self.C.copy()
        
        self.mean = np.zeros(self.x_dim)
        self.mean_o = np.zeros(self.o_dim)
        
        self.Q = np.eye(self.x_dim)
        self.R = np.eye(self.o_dim)
        
    def forward_approx(self):
        en_num = self.en_num
        for i,s in enumerate(self.sigmas):
            self.sigmas[i] = self.fx(s,self.dt)
        
        e = multivariate_normal(self.mean, self.Q, en_num)
        self.sigmas += e
        
        self.x = np.mean(self.sigmas, axis = 0)
        self.C = outer_product_sum(self.sigmas - self.x) / (en_num - 1)
        