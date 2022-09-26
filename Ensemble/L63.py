# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:38:22 2022

@author: Bryan
"""
import numpy as np
import matplotlib.pyplot as plt
from ENKF import EnsembleKalmanFilter

def createList(r1, r2, dt):
    if (r1 == r2):
        return r1
  
    else:
  
        # Create empty list
        res = []
  
        # loop to append successors to 
        # list until r2 is reached.
        while(r1 < r2+0.01):
              
            res.append(r1)
            r1 += dt
        return res
    
def fx(x, gam, rho, bet, dt):
    x = x[0] + ( gam*(x[1]-x[0]) ) * dt
    y = x[1] + ( rho*x[0] - x[1] - x[0]*x[2] ) * dt
    z = x[2] + ( x[0]*x[1] - bet*x[2]) * dt
    
    return [x,y,z]

def hx(x):
   return np.array([x[0]])

N = 4000
dt = 0.01
dtobs = 0.2
tobs = createList(0, 40, dtobs)
t = createList(0, 40, dt)
x_t = np.zeros(N+1)
y_t = np.zeros(N+1)
z_t = np.zeros(N+1)
x_t[0] = 1.508870
y_t[0] = -1.531271
z_t[0] = 25.46091
x_obs = np.zeros(201)
y_obs = np.zeros(201)
z_obs = np.zeros(201)
x_obs[0] = 1.508870
y_obs[0] = -1.531271
z_obs[0] = 25.46091
gam = 10
rho = 28
bet = 8/3
mu = 0
sig = np.sqrt(2);
for i in range(1,N+1):
    x_t[i] = x_t[i-1] + ( gam*(y_t[i-1]-x_t[i-1]) ) * dt;
    y_t[i] = y_t[i-1] + ( rho*x_t[i-1] - y_t[i-1] - x_t[i-1]*z_t[i-1] ) * dt;
    z_t[i] = z_t[i-1] + ( x_t[i-1]*y_t[i-1] - bet*z_t[i-1]) * dt;

for i in range(1,201):
    x_obs[i] = x_t[i*20] + np.random.normal(mu, sig)
    y_obs[i] = y_t[i*20] + np.random.normal(mu, sig)
    z_obs[i] = z_t[i*20] + np.random.normal(mu, sig)
    

fig, ax = plt.subplots(3)

ax[0].scatter(tobs,x_obs,s=3.5)
ax[0].plot(t,x_t,'r--',linewidth=0.7)
ax[0].set_title('(a) xobs')
ax[0].legend(['w noises','w/o noises'],loc='lower left')

ax[1].scatter(tobs,y_obs,s=3.5)
ax[1].plot(t,y_t,'r--',linewidth=0.7)
ax[1].set_title('(b) yobs')
ax[1].legend(['w noises','w/o noises'],loc='lower left')

ax[2].scatter(tobs,z_obs,s=3.5)
ax[2].plot(t,z_t,'r--',linewidth=0.7)
ax[2].set_title('(c) zobs')
ax[2].legend(['w noises','w/o noises'],loc='lower left')

fig.tight_layout()

plt.show()

# Implementing Ensemble Kalman Filter
dim_z = 3
dt = 0.01
En_Num = 500 # Ensemble Number/ Sigma Points
print(np.cov([x_obs, y_obs, z_obs]))
f = EnsembleKalmanFilter(x=[x_t[0], y_t[0], z_t[0]], P=np.cov([x_obs, y_obs, z_obs]), dim_z=dim_z, dt=dt,
                         N=N, hx=hx, fx=fx)

for i in range()
