# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 00:16:21 2022

@author: Bryan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as MN
from numpy.linalg import inv, svd, eig
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
    def __init__(self, x_init, x_dim, o_dim, ens_num, P_p, H, pred, dt):
        self.x = x_init
        self.x_dim = x_dim
        self.o_dim = o_dim
        self.ens_num = ens_num
        self.P_p = P_p
        self.H = H
        self.pred = pred
        self.dt = dt
        self.mean = np.zeros(self.x_dim)
        self.mean_o = np.zeros(self.o_dim)
        self.one = np.ones((ens_num,ens_num))/ens_num
        self.A = MN(self.x, cov = self.P_p, size = self.ens_num).T
        
    def sample_ens(self, mean, cov, size):
        return MN(mean = mean, cov = cov, size = size).T
    def predic(self):
        self.x = self.pred(self.x, self.dt)
        return self.x
    def ENKF(self, o):
        """ 

        Parameters
        ----------
        o : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # setup
        
        
        # self.x = self.pred(self.x, self.dt)
        self.A = self.sample_ens(self.x, self.P_p, self.ens_num)
        
        

        #for i,s in enumerate(self.A.T):
         #   self.A.T[i] = self.pred(s,self.dt)
        Abar = self.A@self.one
        Aprime = self.A-Abar
        self.P_p = Aprime@Aprime.T/(self.ens_num - 1)
        
        D = self.sample_ens(o,np.eye(self.o_dim),self.ens_num)
        gam = D - D@self.one
        Dprime = D - self.H@self.A
        R_e = gam@gam.T/(self.ens_num-1)
        
        # analysis
        the_inv = inv(self.H@self.P_p@self.H.T + R_e)
        #U, Sig, VT = svd(self.H@Aprime + gam, full_matrices = True)
        #diag = []
        #Sig = np.power(Sig, 2)
        #Sigsum = 0
        #Sigsum1 = 0
        #for i in Sig:
        #    Sigsum = Sigsum + i
        #for i in Sig:
        #    if (Sigsum1 / Sigsum) < 0.999:
        #        Sigsum1 = Sigsum1 + i
        #        diag.append(1/i)
        #    else:
        #        diag.append(0)

        #X1 = np.diag(diag)@U.T
        #X2 = X1@Dprime
        #X3 = U@X2
        #X4 = (self.H@Aprime).T@X3
        #X5 = np.eye(self.ens_num) + X4
        X5 = Aprime.T/(self.ens_num-1)@self.H.T@the_inv@(D-self.H@self.A)
        # self.A = self.A + Aprime@X4
        self.A = self.A + self.P_p@self.H.T@the_inv@(D-self.H@self.A)
        # posterior state x
        self.x = np.mean(self.A, axis = 1)
        
        return self.x, self.A, X5
    
    def ENKS(self, AL, X5L, N):
        AsL = []
        xL = []
        for i in range(N):
            Pi = np.eye(self.ens_num)
            for j in range(i+1,N+1):
                Pi = Pi@X5L[j]
            As = AL[i]@Pi
            x = np.mean(As, axis = 1)
            AsL.append(As)
            xL.append(x)
        return xL,AsL
        