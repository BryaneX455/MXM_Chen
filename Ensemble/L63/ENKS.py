# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 00:16:21 2022

@author: Bryan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as MN
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
class ENKS:
    def init(self, x_init, x_dim, o_dim, ens_num, P_p, P_o, H):
        self.x = x_init
        self.x_dim = x_dim
        self.o_dim = o_dim
        self.ens_num = ens_num
        self.P_p = P_p
        self.P_o = P_o
        self.H = H
        
    def ENKF(self):
        x_ens = MN(self.x, cov = self.P_p, size = self.ens_num)
        for i in range(len(x_ens)):
            x_ens[i] = 