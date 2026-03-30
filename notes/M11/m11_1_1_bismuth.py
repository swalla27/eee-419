# radioactive decay example
# author: allee updated by sdm

import numpy as np                # import packages
import matplotlib.pyplot as plt
from random import random

# constants
TAU_BI213 = 46.0*60               # half life bismuth213 in s
TAU_TL = 2.2*60                   # half life thallium in s
TAU_PB = 3.3*60                   # half life lead in s
P_BI2PB = 0.9791                  # probability that Bi decays to Pb vs Tl

h = 1.0                           # size of time step in s
tmax = 20001                      # total time to simulate, in s

p_bi213 = 1.0-2.0**(-h/TAU_BI213) # probability of bismuth decay in one step
p_tl = 1.0-2.0**(-h/TAU_TL)       # probability of thallium decay in one step
p_pb = 1.0-2.0**(-h/TAU_PB)       # probability of lead decay in one step

n_bi213 = 10000                   # number of initial bismuth213 atoms
n_tl = 0                          # number of initial thallium atoms
n_pb = 0                          # number of initial lead atoms
n_bi209 = 0                       # number of initial bismuth209 atoms

t_points = np.arange(0.0,tmax,h)  # array of times 

bi213_points = np.zeros_like(t_points)   # create arrays to hold the data
tl_points    = np.zeros_like(t_points)
pb_points    = np.zeros_like(t_points)
bi209_points = np.zeros_like(t_points)

for t in range(tmax):             # for each time step...
    bi213_points[t] = n_bi213     # insert the current value into its array
    tl_points[t]    = n_tl
    pb_points[t]    = n_pb
    bi209_points[t] = n_bi209
    
    # calculate the number of Pb that decay - last to decay so need to do first!
    decay = 0                     # count of decayed atoms
    for i in range(n_pb):         # for each of the atoms...
        if random()<p_pb:         # if it has decayed
            decay += 1            # increment the counter

    n_pb -= decay                 # remove the decayed atoms from lead
    n_bi209 += decay              # and add to stable bismuth
    
    # calculate the number of Tl that decay
    decay = 0                     # count of decayed atoms
    for i in range(n_tl):         # for each of the atoms...
        if random()<p_tl:         # if it has decayed
            decay += 1            # increment the counter

    n_tl -= decay                 # remove the decayed from thalium
    n_pb += decay                 # and add to lead
    
    # calculate the number of Bi213 that decay
    decay_a = 0                   # count of decayed atoms
    decay_b = 0                   # count of decayed atoms
    for i in range(n_bi213):      # for each of the atoms...
        if random()<p_bi213:      # if it has decayed
            if random()<P_BI2PB:  # decaying to lead or thalium?
                decay_a += 1      # imcrement count of those to lead
            else:
                decay_b += 1      # imcrement count of those to thalium

    n_bi213 -= decay_a + decay_b  # remove the decayed from unstable bismuth
    n_pb += decay_a               # and add some to lead
    n_tl += decay_b               # and the rest to thalium
    
# plot just the bismuth
plt.plot(t_points,bi213_points,'red',label="bismuth 213")
plt.plot(t_points,bi209_points,'blue',label="bismuth 209")
plt.legend()
plt.xlabel('time')
plt.ylabel('# of atoms')
plt.title('Bismuth 213 to Bismuth 209')
plt.show()

# now thalium and lead
plt.plot(t_points,tl_points,'orange',label="thalium")
plt.plot(t_points,pb_points,'green',label="lead")
plt.legend()
plt.xlabel('time')
plt.ylabel('# of atoms')
plt.title('Lead vs Thallium')
plt.show()

# now all 4 to show scale
plt.plot(t_points,bi213_points,'red',label="bismuth 213")
plt.plot(t_points,bi209_points,'blue',label="bismuth 209")
plt.plot(t_points,tl_points,'orange',label="thalium")
plt.plot(t_points,pb_points,'green',label="lead")
plt.legend()
plt.xlabel('time')
plt.ylabel('# of atoms')
plt.title('Bismuth Decay, All Isotopes')
plt.show()

