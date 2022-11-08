# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:11:00 2022

@author: 16088
"""
import numpy as np
import matplotlib.pyplot as plt
from ENKS import ENKS

N = 4000
dt = 0.01
dtobs = 0.1

gam = 10
rho = 28
bet = 8/3
mu = 0
var = 0
sig = np.sqrt(var)
err_name = '0M' + str(var) + 'V'
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
ratio = int(dtobs/dt)
t = createList(0, 40, dt)
tobs = createList(0, 40, dtobs)
Nobs = len(tobs) - 1 
def fx(x,dt):
    xn = x[0] + ( gam*(x[1]-x[0]) ) * dt
    yn = x[1] + ( rho*x[0] - x[1] - x[0]*x[2] ) * dt
    zn = x[2] + ( x[0]*x[1] - bet*x[2]) * dt
    
    return [xn,yn,zn]

def t2o(x):
    xo = x[0] + np.random.normal(mu, sig)
    yo = x[1] + np.random.normal(mu, sig)
    zo = x[2] + np.random.normal(mu, sig)
    return np.array([xo,yo,zo])


x_t = np.zeros(N+1)
y_t = np.zeros(N+1)
z_t = np.zeros(N+1)
x_t[0] = 1.508870
y_t[0] = -1.531271
z_t[0] = 25.46091

x_obs = np.zeros(Nobs+1)
y_obs = np.zeros(Nobs+1)
z_obs = np.zeros(Nobs+1)
x_obs[0] = 1.508870 + np.random.normal(mu, sig)
y_obs[0] = -1.531271 + np.random.normal(mu, sig)
z_obs[0] = 25.46091 + np.random.normal(mu, sig)


for i in range(1,N+1):
    x_t[i] = x_t[i-1] + ( gam*(y_t[i-1]-x_t[i-1]) ) * dt 
    y_t[i] = y_t[i-1] + ( rho*x_t[i-1] - y_t[i-1] - x_t[i-1]*z_t[i-1] ) * dt 
    z_t[i] = z_t[i-1] + ( x_t[i-1]*y_t[i-1] - bet*z_t[i-1]) * dt 
Obs_ind = []
for i in range(1,Nobs+1):
    x_obs[i] = x_t[(i-1)*ratio + 1] + np.random.normal(mu, sig)
    y_obs[i] = y_t[(i-1)*ratio + 1] + np.random.normal(mu, sig)
    z_obs[i] = z_t[(i-1)*ratio + 1] + np.random.normal(mu, sig)
    Obs_ind.append((i-1)*ratio + 1)

x_dim = 3
o_dim = 3
ens_num = 25
en_num_str = str(ens_num) + '_'

f = ENKS(x_init=[x_t[0], y_t[0], z_t[0]], x_dim = x_dim, o_dim = o_dim, ens_num = ens_num, P_p = np.eye(x_dim), H = np.eye(o_dim), pred = fx, dt = dt)
xiL = []
AL = []
X5L = []
for i in range(N):
    if i in Obs_ind:
        z = [x_obs[int((i-1)/ratio+1)], y_obs[int((i-1)/ratio+1)],z_obs[int((i-1)/ratio+1)]]
        xi, A, X5= f.ENKF(np.asarray(z))
        AL.append(A)
        X5L.append(X5)
    else:
        xi = f.predic()# [x_t[i+1], y_t[i+1], z_t[i+1]] #f.predic()
    xiL.append(xi) 
xl = []
yl = []
zl = []
for i in range(N):
   xl.append(xiL[i][0])
   yl.append(xiL[i][1])
   zl.append(xiL[i][2])
""" 
xsL, AsL = f.ENKS(AL,X5L,Nobs)
xsL.append(xiL[N])
print(len(xsL))
print(len(xsL[0]))

xls = []
yls = []
zls = []
for i in range(N+1):
   xls.append(xsL[i][0])
   yls.append(xsL[i][1])
   zls.append(xsL[i][2])
"""

fig, ax = plt.subplots(3)

ax[0].plot(t,x_t,'r--',linewidth=0.7)
ax[0].plot(t[:-1],xl,'g',linewidth=0.7)
#ax[0].plot(t,xls,'b', linewidth=0.7)
ax[0].set_title('(a) x_true, x_smoother_En' + str(ens_num))

ax[1].plot(t,y_t,'r--',linewidth=0.7)
ax[1].plot(t[:-1],yl,'g',linewidth=0.7)
#ax[1].plot(t,yls,'b',linewidth=0.7)
ax[1].set_title('(b) y_true_, y_smoother_En' + str(ens_num))

ax[2].plot(t,z_t,'r--',linewidth=0.7)
ax[2].plot(t[:-1],zl,'g',linewidth=0.7)
# ax[2].plot(t,zls,'b',linewidth=0.7)
ax[2].set_title('(c) z_true, z_smoother_En' + str(ens_num))

ax[2].legend(['w/o noises', 'filt'], loc = 'lower left')
fig.tight_layout()

plt.show()
plt.savefig('L63_' + str(o_dim) + 'observation' + str(ens_num) + 'ensemble.eps', format='eps')

