# solving the 3-body problem for stars
# author: allee updated by sdm

import numpy as np                    # for arrays
import matplotlib.pyplot as plt       # plotting
from scipy.integrate import odeint    # solving differential eqn

# the vector is yv = [x1,y1,x2,y2,x3,y3,xx1,yy1,xx2,yy2,xx3,yy3]
#                     0  1  2  3  4  5  6   7   8   9   10  11

m1=150   # define the masses of the stars
m2=200
m3=250
G=1      # define the gravitional constant to be easy

################################################################################
# Function to return the derivative of the vector.                             #
# Inputs:                                                                      #
#    yv - the vector as defined above                                          #
#    time - the current time                                                   #
# Outputs:                                                                     #
#     the vector of derivatives                                                #
################################################################################

def calc_derivative(yv, tv):
    dx1dt=yv[6]               # the first six entries get copied from the last
    dy1dt=yv[7] 
    dx2dt=yv[8]
    dy2dt=yv[9]
    dx3dt=yv[10]
    dy3dt=yv[11]

    # calculate the intermediate values
    d21= ((yv[2]-yv[0])**2 + (yv[3]-yv[1])**2)**1.5
    d31= ((yv[4]-yv[0])**2 + (yv[5]-yv[1])**2)**1.5
    d32= ((yv[4]-yv[2])**2 + (yv[5]-yv[3])**2)**1.5

    # now calculate the last six derivatives
    dxx1dt=G*m2*(yv[2]-yv[0])/d21 + G*m3*(yv[4]-yv[0])/d31
    dyy1dt=G*m2*(yv[3]-yv[1])/d21 + G*m3*(yv[5]-yv[1])/d31
    dxx2dt=G*m1*(yv[0]-yv[2])/d21 + G*m3*(yv[4]-yv[2])/d32
    dyy2dt=G*m1*(yv[1]-yv[3])/d21 + G*m3*(yv[5]-yv[3])/d32
    dxx3dt=G*m1*(yv[0]-yv[4])/d31 + G*m2*(yv[2]-yv[4])/d32
    dyy3dt=G*m1*(yv[1]-yv[5])/d31 + G*m2*(yv[3]-yv[5])/d32

    #returning the derivatives of each
    return (dx1dt,dy1dt,dx2dt,dy2dt,dx3dt,dy3dt,
            dxx1dt,dyy1dt,dxx2dt,dyy2dt,dxx3dt,dyy3dt)

yinit = (3,1,-1,-2,-1,1,0,0,0,0,0,0)           # initial values
tvec = np.linspace(0, 2, 1000)                 # time steps
yvec = odeint(calc_derivative, yinit, tvec)    # solve

plt.plot(yvec[:,0],yvec[:,1],'b',label='small')  # now plot the stars' positions
plt.plot(yvec[:,2],yvec[:,3],'r',label='medium')
plt.plot(yvec[:,4],yvec[:,5],'g',label='large')
plt.xlabel('x')
plt.ylabel('y',rotation=0)
plt.title('Three Body Problem')
plt.legend()
plt.show()
