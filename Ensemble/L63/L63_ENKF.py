# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 00:43:41 2022

@author: Bryan
"""

import numpy as np
import matplotlib.pyplot as plt
from ENKF import ENKF


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
    xo = x[0] + np.random.normal(0, sig)
    yo = x[1] + np.random.normal(0, sig)
    # zo = x[2] + np.random.normal(0, sig)
    return np.array([xo,yo])

# Time step number, size
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


t = createList(0, 40, dt)
x_t = np.zeros(N+1)
y_t = np.zeros(N+1)
z_t = np.zeros(N+1)
x_t[0] = 1.508870
y_t[0] = -1.531271
z_t[0] = 25.46091

for i in range(1,N+1):
    x_t[i] = x_t[i-1] + ( gam*(y_t[i-1]-x_t[i-1]) ) * dt;
    y_t[i] = y_t[i-1] + ( rho*x_t[i-1] - y_t[i-1] - x_t[i-1]*z_t[i-1] ) * dt;
    z_t[i] = z_t[i-1] + ( x_t[i-1]*y_t[i-1] - bet*z_t[i-1]) * dt;
    
    
# Implementing ENKF
o_dim = 3
en_num = 100 # Ensemble Number/ Sigma Points

f = ENKF(x_init=[x_t[0], y_t[0], z_t[0]], x_init_cov=np.cov([x_t, y_t, z_t]), o_dim=o_dim, dt=dt,
                         en_num=en_num, t2o=t2o, f=fx)

xiL = []
#fig, ax = plt.subplots(2,2)
H = np.eye(2)
H = np.insert(H, 2, 0, axis=1)
print(H)
for i in range(N+1):
    xi= f.ENKF(H, var)
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
ax[0].set_title('(a) x_true, x_filt_En' + str(en_num))

ax[1].plot(t,y_t,'r--',linewidth=0.7)
ax[1].plot(t,yl,'g',linewidth=0.7)
ax[1].set_title('(b) y_true_, y_filt_En' + str(en_num))

ax[2].plot(t,z_t,'r--',linewidth=0.7)
ax[2].plot(t,zl,'g',linewidth=0.7)
ax[2].set_title('(c) z_true, z_filt_En' + str(en_num))

ax[2].legend(['w/o noises', 'filt'], loc = 'lower left')
fig.tight_layout()

plt.show()
# fig.savefig('L63_EN'+ en_num_str + err_name +'.eps', format='eps')

