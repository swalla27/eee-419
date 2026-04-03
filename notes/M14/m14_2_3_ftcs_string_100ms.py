# Simulate a piano string
# author: allee updated by sdm

import numpy as np                  # import packages
import matplotlib.pyplot as plt

# Constants
L = 1.0                             # string length in m
D = 0.1                             # string hit this distance from end
C = 1.0                             # m/s
SIG = 0.3                           # m
V = 100.0                           # m/s
N = 100                             # number of spatial steps

# derived values
h = 1e-6                            # s, time step
epsilon = h/1000.0                  # small value for float comparisons
a = L/N                             # space between steps
k1 = C / ( L * L )                  # precompute this constant
k2 = -2.0*SIG*SIG                   # precompute this constant
k3 = h * V * V / ( a * a )          # precompute this constant

# times at which to plot the results
t1 = 5.0e-3                         # 5ms
t2 = 37.0e-3
t3 = 40.0e-3 
t4 = 80.0e-3
t5 = 100.0e-3
tend = t5 + epsilon                 # make sure we simulate through t5

# create arrays

phi = np.zeros(N+1,float)           # phi is displacement
phin = np.zeros(N+1,float)          # phi next
psi = np.zeros(N+1,float)           # psi is velocity
psin = np.zeros(N+1,float)          # psi next

# create array of initial velocities
for i in range(0,N+1):              # x is i*a
    x = i*a                         # how far along the string...
    psi[i] = k1 * x * (L-x) * np.exp( (x-D) * (x-D) / k2)

# main loop
t = 0.0
psi_snaps = []                      # track psi and phi for plotting
psi_snaps.append(psi.copy())        # NOTE: use copy to save the values
phi_snaps = []
phi_snaps.append(phi.copy())

while t < tend:                       # for each time step...
    #calculate new values of phi and psi; ends don't move!
    phin[1:N] = phi[1:N] + h*psi[1:N]
    psin[1:N] = psi[1:N] + k3*(phi[2:N+1]+phi[0:N-1]-2*phi[1:N])

    phi,phin = phin,phi               # swap them back
    psi,psin = psin,psi
    
    if abs(t-t1)<epsilon:
        psi_snaps.append(psi.copy())  # save the values for plotting
        phi_snaps.append(phi.copy())
    elif abs(t-t2)<epsilon:
        psi_snaps.append(psi.copy())
        phi_snaps.append(phi.copy())
    elif abs(t-t3)<epsilon:
        psi_snaps.append(psi.copy())
        phi_snaps.append(phi.copy())
    elif abs(t-t4)<epsilon:
        psi_snaps.append(psi.copy())
        phi_snaps.append(phi.copy())
    elif abs(t-t5)<epsilon:
        psi_snaps.append(psi.copy())
        phi_snaps.append(phi.copy())
        
    t += h                            # next time step

colors = ['black','red','orange','magenta','green','blue']

for index,color in enumerate(colors):
    plt.plot(1000*phi_snaps[index],color,label="t"+str(index))

plt.xlabel("x")
plt.ylabel("1000*phi")
plt.title("Piano String Displacement Up to 100ms")
plt.legend()
plt.show()

for index,color in enumerate(colors):
    plt.plot(psi_snaps[index],color,label="t"+str(index))
plt.xlabel("x")
plt.ylabel("psi")
plt.title("Piano String Velocity Up to 100ms")
plt.legend()
plt.show()
