# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 23:49:39 2022

@author: Bryan
"""
import numpy as np
from numpy.random import multivariate_normal as MN
from helpers import outer_product_sum

class ENKF():
    def __init__(self, x_init, x_init_cov, o_dim, dt, en_num, t2o, f):
        self.x = x_init
        self.C = x_init_cov
        self.o_dim = o_dim
        self.dt = dt
        self.en_num = en_num
        self.t2o = t2o
        self.f = f
        self.ens = MN(mean=x_init, cov=x_init_cov, size=self.en_num)
    def ENKF(self, H, OV):
        en_num = self.en_num
        for i,enp in enumerate(self.ens):
            self.ens[i] = self.f(enp,self.dt)
        self.x = np.mean(self.ens, axis = 0)
        e = np.ones((1,en_num))
        A = self.ens.T - self.x.T.reshape((3,1))@e
        self.C = A@A.T/(en_num-1)
        
        ens_o = np.zeros((en_num, self.o_dim))
        OC = np.eye(self.o_dim) * OV
        for i in range(en_num):
            ens_o[i] = self.t2o(self.ens[i])
        
        o_mean = np.mean(ens_o, axis=0)
        C_zz = (outer_product_sum(ens_o - o_mean) / (en_num-1)) + OC
        C_xz = outer_product_sum(self.ens - self.x, ens_o - o_mean) / (en_num - 1)
        self.S = C_zz
        self.SI = np.linalg.inv(self.S)
        self.KG = np.dot(C_xz, self.SI)
        self.ens = (self.ens.T + self.KG@(ens_o.T - H@self.ens.T)).T 
        self.x = np.mean(self.ens, axis = 0)

        
        return self.x
        
        
        
        