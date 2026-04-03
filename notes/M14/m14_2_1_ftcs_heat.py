# Example 9.3 in Newman
# author: allee updated by sdm

import numpy as np              # import packages
import matplotlib.pyplot as plt
import matplotlib.cm as cm      # color map

# Constants
L = 0.01                        # thickness of the pot
D = 4.25e-6                     # diffusion coefficient
N = 100                         # number of steps

a = L/N                         # grid step size
h = 1e-4                        # time step size
epsilon = h/1000                # tolerance for float compares

t_cold = 0.0                    # cold water temperature (inside pot)
t_pot = 20.0                    # initial temperature of the pot
t_hot = 50.0                    # hot water temperature (outside pot)

t1 = 0.01                       # times at which to plot temperature gradient
t2 = 0.1
t3 = 0.4
t4 = 1.0
t5 = 10.0
tend = t5 + epsilon             # time to end the simulation

#create arrays to model a vertical slice through the bottom of the pot
T = np.full(N+1,t_pot,float)    # init all values to the temp of the pot
T[0] = t_hot                    # except inside temp is cold
T[N] = t_cold                   # and outside temp is hot

Tp = np.copy(T)                 # make identical since never adjust 0 and N

# main loop
t = 0.0                         # start at time 0
c = h*D/(a*a)                   # calculate the required coefficient
plt.plot(T,label='t0='+str(0)+'s')
while t < tend:                 # can do entire array all at once!
    Tp[1:N] = T[1:N] + c * ( T[0:N-1] + T[2:N+1] - ( 2 * T[1:N] ) )

    T,Tp = Tp,T      # swap them
    
    # Note how comparisons are done to check progress
    if abs(t-t1) < epsilon:     # make plots at the given time
        plt.plot(T,label='t1='+str(t1)+'s')
    elif abs(t-t2) < epsilon:
        plt.plot(T,label='t2='+str(t2)+'s')
    elif abs(t-t3) < epsilon:
        plt.plot(T,label='t3='+str(t3)+'s')
    elif abs(t-t4) < epsilon:
        plt.plot(T,label='t4='+str(t4)+'s')
    elif abs(t-t5) < epsilon:
        plt.plot(T,label='t5='+str(t5)+'s')
        
    t += h                      # go to next time

plt.xlabel("x")                 # and show the results
plt.ylabel("T",rotation=0)
plt.title('Heat Flow Using FTCS')
plt.legend()
plt.show()
