# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 00:16:21 2022

@author: Bryan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as MN
from numpy.linalg import inv
"""
---------------------------------------------------------------------------------------------------------------------
Overview:
    This class is a combination of ensemble kalman filter and ensemble kalman smoother. Good for large state space
---------------------------------------------------------------------------------------------------------------------
Parameters:
    x_init: 
        (x_dim,1)
    ens_num:
        number of ensemble points for model states and observation data
        the error of Monte Carlo sampling will decrease in order of 1/sqrt(ens_num)

    P_p:
        (x_dim,x_dim)
        covariance of prediction step
        
    P_a:
        (x_dim,x_dim)
        covariance of analysis step
        
    P_o:
        (o_dim,o_dim)
        covariance of observation step
        
    H:
        (o_dim,x_dim)
        linear transformation from model to observation
        
    K:
        Kalman gain matrix, derived by minimizing the traces of prediction covariance matrix P_p
        
    
"""
class ENKS():
    def __init__(self, x_init, x_dim, o_dim, ens_num, P_p, P_o, H, t2o, pred, dt):
        self.x = x_init
        self.x_dim = x_dim
        self.o_dim = o_dim
        self.ens_num = ens_num
        self.P_p = P_p
        self.P_o = P_o
        self.H = H
        self.t2o = t2o
        self.pred = pred
        self.dt = dt
        self.x_ens = MN(self.x, cov = self.P_p, size = self.ens_num)
        self.o_ens = np.zeros((ens_num,o_dim))
        self.mean = np.zeros(self.x_dim)
        self.mean_o = np.zeros(self.o_dim)
    def cov(self,A,B):
        C = 0
        for i in range(len(A)):
            ctemp = np.outer(A[i], B[i])
            C += ctemp
            
        return C
        
    def ENKF(self, o):
        # predict state ensemble and find its prediction covariance matrix
        for i,ee in enumerate(self.x_ens):
            self.x_ens[i] = self.pred(ee,self.dt)
        self.x = np.mean(self.x_ens, axis = 0)
        self.P_p = self.cov(self.x_ens-self.x, self.x_ens-self.x)/(self.ens_num-1)        
        
        # data ensemble construction
        for i in range(self.ens_num):
            self.o_ens[i] = self.t2o(self.x_ens[i])   
        o_mean = np.mean(self.o_ens, axis = 0)
        P_oe = np.eye(self.o_dim)*10
        
        # perform the analysis/filtering step, construct the analyzed state covariance matrix
        K = self.P_p@self.H.T@(inv(self.H@self.P_p@self.H.T + P_oe))
        for i,ee in enumerate(self.x_ens):
            self.x_ens[i] = ee + K@(self.o_ens[i] - self.H@ee)
            # if observation dimension is larger than the ensemble number, the first part in the bracket will be singular.
        
        self.P_p = self.P_p - K@self.H@self.P_p
        
        # take the mean of the posterior enssemble as posterior state
        self.x = np.mean(self.x_ens, axis = 0)
        return self.x, self.P_p,self.x_ens