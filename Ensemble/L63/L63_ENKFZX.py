# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:38:22 2022

@author: Bryan
"""
import numpy as np
import matplotlib.pyplot as plt
from ENKF_ZX import Ensemble_Kalman_Filter


def createList(r1, r2, dt):
    if (r1 == r2):
        return r1
  
    else:
  
        # Create empty list
        res = []
  
        # loop to append successors to 
        # list until r2 is reached.
        while(r1 < r2+dt):
              
            res.append(r1)
            r1 += dt
        return res
    
def fx(x,dt):
    xn = x[0] + ( 10*(x[1]-x[0]) ) * dt
    yn = x[1] + ( 28*x[0] - x[1] - x[0]*x[2] ) * dt
    zn = x[2] + ( x[0]*x[1] - 8/3*x[2]) * dt
    
    return [xn,yn,zn]

def t2o(x):
    xo = x[0] + np.random.normal(mu, sig)
    yo = x[1] + np.random.normal(mu, sig)
    # zo = x[2] + np.random.normal(mu, sig)
    return np.array([xo,yo])

N = 4000
dt = 0.01
dtobs = 0.2
tobs = createList(0, 40, dtobs)

gam = 10
rho = 28
bet = 8/3
mu = 0
var = 100
sig = np.sqrt(var)
err_name = '0M' + str(var) + 'V'

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
x_obs[0] = 1.508870 + np.random.normal(mu, sig)
y_obs[0] = -1.531271 + np.random.normal(mu, sig)
z_obs[0] = 25.46091 + np.random.normal(mu, sig)

x_obs_full = np.zeros(N+1)
y_obs_full = np.zeros(N+1)
z_obs_full = np.zeros(N+1)
x_obs_full[0] = 1.508870 + np.random.normal(mu, sig)
y_obs_full[0] = -1.531271 + np.random.normal(mu, sig)
z_obs_full[0] = 25.46091 + np.random.normal(mu, sig)


for i in range(1,N+1):
    x_t[i] = x_t[i-1] + ( gam*(y_t[i-1]-x_t[i-1]) ) * dt;
    y_t[i] = y_t[i-1] + ( rho*x_t[i-1] - y_t[i-1] - x_t[i-1]*z_t[i-1] ) * dt;
    z_t[i] = z_t[i-1] + ( x_t[i-1]*y_t[i-1] - bet*z_t[i-1]) * dt;
    
for i in range(1,N+1):
    x_obs_full[i] = x_t[i] + np.random.normal(mu, sig)
    y_obs_full[i] = y_t[i] + np.random.normal(mu, sig)
    # z_obs_full[i] = z_t[i] + np.random.normal(mu, sig)
    
for i in range(1,201):
    x_obs[i] = x_t[i*20] + np.random.normal(mu, sig)
    y_obs[i] = y_t[i*20] + np.random.normal(mu, sig)
    z_obs[i] = z_t[i*20] + np.random.normal(mu, sig)
    

fig, ax = plt.subplots(3)

ax[0].scatter(tobs,x_obs,s=3.5)
ax[0].plot(t,x_t,'r--',linewidth=0.7)
ax[0].set_title('(a) x_true, x_obs')
ax[0].legend(['w noises','w/o noises'],loc='lower left')

ax[1].scatter(tobs,y_obs,s=3.5)
ax[1].plot(t,y_t,'r--',linewidth=0.7)
ax[1].set_title('(b) y_true_, y_obs')
ax[1].legend(['w noises','w/o noises'],loc='lower left')

ax[2].scatter(tobs,z_obs,s=3.5)
ax[2].plot(t,z_t,'r--',linewidth=0.7)
ax[2].set_title('(c) z_true, z_obs')
ax[2].legend(['w noises','w/o noises'],loc='lower left')

fig.tight_layout()

plt.show()
fig.savefig('L63_TO_'+ err_name +'.eps', format='eps')


# Implementing Ensemble Kalman Filter
o_dim = 2
dt = 0.01
En_Num = 100 # Ensemble Number/ Sigma Points
en_num_str = str(En_Num) + '_'

# ENKF 
f = Ensemble_Kalman_Filter(x_init=[x_t[0], y_t[0], z_t[0]], x_init_cov=np.eye(3), o_dim=o_dim, dt=dt,
                         en_num=En_Num, t2o=t2o, fx=fx)
xiL = []
#fig, ax = plt.subplots(2,2)
for i in range(N+1):
    z = [x_obs_full[i], y_obs_full[i]]
    f.forward_approx()
    xi, K = f.enks_filter(np.asarray(z))
    xiL.append(xi) 
xl = []
yl = []
zl = []
for i in range(N+1):
   xl.append(xiL[i][0])
   yl.append(xiL[i][1])
   zl.append(xiL[i][2])
   
fig, ax = plt.subplots(3)

ax[0].plot(t,x_t,'r--',linewidth=0.7)
ax[0].plot(t,xl,'g',linewidth=0.7)
ax[0].set_title('(a) x_true, x_filt_En' + str(En_Num))

ax[1].plot(t,y_t,'r--',linewidth=0.7)
ax[1].plot(t,yl,'g',linewidth=0.7)
ax[1].set_title('(b) y_true_, y_filt_En' + str(En_Num))

ax[2].plot(t,z_t,'r--',linewidth=0.7)
ax[2].plot(t,zl,'g',linewidth=0.7)
ax[2].set_title('(c) z_true, z_filt_En' + str(En_Num))

ax[2].legend(['w/o noises', 'filt'], loc = 'lower left')
fig.tight_layout()

plt.show()
fig.savefig('L63_EN'+ en_num_str + err_name + str(o_dim) + 'obs' + '.eps', format='eps')


