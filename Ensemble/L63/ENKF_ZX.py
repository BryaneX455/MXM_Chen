# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:15:03 2022

@author: Bryan
"""
import numpy as np
from numpy.random import multivariate_normal
from helpers import outer_product_sum
from copy import deepcopy
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
        self.o_dim = o_dim
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
        
        self.sigmas = multivariate_normal(mean=x_init, cov=x_init_cov, size=self.en_num)
        
    def forward_approx(self):
        en_num = self.en_num
        for i,s in enumerate(self.sigmas):
            self.sigmas[i] = self.fx(s,self.dt)
        
        e = multivariate_normal(self.mean, self.Q, en_num)
        self.sigmas += e
        
        self.x = np.mean(self.sigmas, axis = 0)
        self.C = outer_product_sum(self.sigmas - self.x) / (en_num - 1)
        
        # Calculate posterior
        

    def enks_filter(self, o, R = None):
        # Check if the observation is empty
        if o is None:
            self.o = np.array([[None]*self.o_dim]).T
            self.x_post = self.x.copy()
            self.C_post = self.C.copy()
            return
        
        if R is None:
            R = self.R
        if np.isscalar(R):
            R = np.eye(self.dim_z) * R
            
        en_num = self.en_num
        o_dim = len(o)
        sigmas_h = np.zeros((en_num, o_dim))
        
        for i in range(en_num):
            sigmas_h[i] = self.t2o(self.sigmas[i])
            
        o_mean = np.mean(sigmas_h, axis=0)
        
        # The first part is the covariance matrix of the data matrix covariance matrix 
        C_zz = (outer_product_sum(sigmas_h - o_mean) / (en_num-1)) + R
        C_xz = outer_product_sum(self.sigmas - self.x, sigmas_h - o_mean) / (en_num - 1)


        self.S = C_zz
        self.SI = np.linalg.inv(self.S)
        self.KG = np.dot(C_xz, self.SI)
        
        e_r = multivariate_normal(self.mean_o, R, en_num)
        for i in range(en_num):
            self.sigmas[i] += np.dot(self.KG, o + e_r[i] - sigmas_h[i])

        self.x = np.mean(self.sigmas, axis=0)
        self.C = self.C - np.dot(np.dot(self.KG, self.S), self.KG.T)

        # save measurement and posterior state
        self.o = deepcopy(o)
        self.x_post = self.x.copy()
        self.C_post = self.C.copy()
        
        return self.x, self.KG
        