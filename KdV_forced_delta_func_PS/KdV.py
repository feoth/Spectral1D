#!/usr/bin/env python
# coding: utf-8

# # Korteweg - de Vries

# $$ u_{t} + \frac{\beta c}{4 E} (u^2)_{\xi} + \frac{\nu^2 R^2 c}{4} u_{\xi\xi\xi} = 0 $$

# In[157]:


import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
from IPython.display import HTML, clear_output, display
from ipywidgets import IntSlider, Output
plt.rcParams['animation.html'] = 'html5'
from log_progress import log_progress


# Domain and initial condition function:

# In[183]:


# m to mm
coord_scale = 1e3
# s to mus
time_scale = 1e6

# domain
t0, tf = 0, 5e-5*time_scale
dt = 1e-5
T = np.linspace(t0, tf, int((tf - t0)/dt + 1))
pN = 1000
xl, xr = -0.04*coord_scale, 0.04*coord_scale
L = xr - xl
X = np.linspace(xl, xr, pN, endpoint=False)

# high frequency range for de-aliasing
hfreq1 = int(np.round(pN/3)) - 1
hfreq2 = int(np.round(2*pN/3)) + 1
da = np.ones_like(X)
da[hfreq1:hfreq2] = 0

# elastic moduli, radius, linear wave velocity
rho = 1.06; E = 3.7; nu = 0.34; l = -18.9; m = -13.3; n = -10.0 # PS
R = 5e-3*coord_scale
c = np.sqrt(E/rho*1e6)*coord_scale/time_scale
v_water = 1570*coord_scale/time_scale
v = c - v_water
P = 872*101325

def stress_tan(x, t):
    V = 0
    D = 0.2
    arg = ((x - V*t) - (xr + xl)/2)/D
    return P*np.cosh(arg)**(-2)
    
# nonlinear and dispersion coefficients
beta = (3*E + 2*l*(1 - 2*nu)**3 + 4*m*(1 + nu)**2*(1 - 2*nu) + 6*n*nu**2)/4/E*c
d = nu**2*R**2/4*c
g = 1/R/(E*1e9)

# soliton parameters
#A = -0.04
#D = np.sqrt(6*d/A/beta)*coord_scale
#V = 2*A*beta/3

def iv(x, t, A, D, V):
    arg = ((x - V*t) - (xr + xl)/2)/D
    u = A*np.cosh(arg)**(-2)
    return u

# initial impulse parameters
D1  = (xr - xl)/500
As1 = 0.000003
A1  = -1/2/D1*coord_scale*As1
V   = 0.0
D2  = (xr - xl)/40
As2 = 0.5
A2  = -1/2/D2*coord_scale*As1*As2 

def complex_iv(x, t):
    return iv(x - 1.65*D2, t, A1, D1, V) - iv(x, t, A2, D2, V)

def simple_iv(x, t):
    return iv(x, t, A1, D1, V)

print("Beta:                  ", beta)
print("Length parameter (mm):  ", D1)
print("Linear velocity:        ", 0)
print("IV velocity:            ", V)
print("Sound velocity (mm/\u03BCs): ", c)

fig, ax = plt.subplots(1, 2, figsize=(12,4))
plt.subplot(121)
plt.title('Initial shock with extension')
plt.ylim(-0.2*np.abs(A1), 1.2*np.abs(A1))
plt.plot(X, -complex_iv(X, 0))
plt.subplot(122)
plt.title('Initial shock')
plt.ylim(-0.2*np.abs(A1), 1.2*np.abs(A1))
plt.plot(X, -simple_iv(X, 0))
plt.show()


# In[184]:


plt.plot(X, stress_tan(X, 0)/P)


# In[185]:


dt


# Derivative function

# In[186]:


# normalize frequencies
xi = np.fft.fftfreq(len(X))*(len(X))*2*np.pi /(X[-1] - X[0])

# sponge
def sponge_coef_func(x):
    amplt = np.abs(A1)*10000
    kappa = 6e-3*L
    lfrac = 3.0/4.0
    return amplt*( 1 + np.tanh(kappa*(x - lfrac*L/2)) 
                  +1 - np.tanh(kappa*(x + lfrac*L/2)))

sp_coef = sponge_coef_func(X)
plt.plot(X, sp_coef)
plt.show()

def der(t, u):
    u_orig = np.fft.ifft(u)
    u_sp = np.fft.fft(u_orig * sp_coef)
    u2 = np.fft.fft(u_orig**2)
    s = np.fft.fft(stress_tan(X, t))
    du = 1j*xi*(-v*u - beta*u2 + d*xi**2*u) - g*s - u_sp
    return du


# Solve and plot

# In[187]:


# de-aliased initial condition
sol = simple_iv(X, 0)
uk = np.fft.fft(sol)*da
# containers for solution and time
U = [sol,]
T_saved = [t0,]


# In[190]:


out = Output()
display(out)

save_rate = 10000
for i in log_progress(range(1, len(T)), every=save_rate):
    sln = sp.integrate.solve_ivp(der, (0, dt), uk, method='RK45')
    # extract solution
    u = sln.y[:, -1]
    # dealiasing
    u *= da
    # save solution at each n-th time-point
    if not i%save_rate:
        U.append(np.fft.ifft(u).real)
        T_saved.append(T[i])
        # plot solution
        out.clear_output(wait=True)
        with out:
            plt.plot(X, -U[-1])
            plt.show()
            print("time:     %.2f \u03BCs" % T_saved[-1])
            dist = v_water*T_saved[-1]
            print("distance: %.2f mm" % dist)
    # define next initial condition for integrator
    uk = u


# In[191]:


direct = "KdV_forced_delta_func_PS/"
np.save(direct + "U.npy", U)
np.save(direct + "T.npy", T_saved)
np.save(direct + "X.npy", X)


# In[100]:


fig, axes = plt.subplots(figsize=(8, 4))
plt.title('Soluton animation')
plt.xlim(X[0], X[-1]) 
plt.ylim(-0.5*np.abs(A1)*1e-1, 1.5*np.abs(A1)*1e-1)
plt.grid()
plt.xlabel("x, mm")
plt.ylabel("deform")
line1, = axes.plot([], [], lw=2)
line2, = axes.plot([], [], lw=2)
time_template = 'time = %.0f'
time_text = axes.text(0.85, 0.9, '', transform=axes.transAxes)
plt.minorticks_on()
plt.close()
dx = (X[-1] - X[0]) / pN

def plot_frame(i):
    axes.set_xlim(X[0] + c*T_saved[i]*0, X[-1]+ c*T_saved[i]*0)
    line1.set_data(X + c*T_saved[i]*0, -U[i])
    time_text.set_text(time_template % T_saved[i])
    fig.canvas.draw()
    return fig

matplotlib.animation.FuncAnimation(fig, plot_frame, frames=len(U), interval=100, repeat=False)


# In[ ]:




