##################
# ##### Code 1: dx(t)/dt is a function in t
# import numpy as np                   # import packages
# import matplotlib.pyplot as plt

# #### x(t) = t**2
# #### dx/dt = 2t
# #### Euler's method x(t+h) = x(t) + hf(x,t)

# #### Define the first derivative of x(t):
# def ode_ex(t):
#     return(2*t)

# a = 0                        # set up the bounds for the problem; from t=a...
# b = 10                       # to t=b...
# N = 1000                     # want 1000 steps
# h = (b-a)/N                  # and this is the step size
# x = 0                        # initial value x(a)=0

# tpoints = np.arange(a,b,h)   # create the array of steps
# xpoints = []                 # the empty list of results

# for t in tpoints:            # at each of the steps
#     xpoints.append(x)        # put in the current value
#     x += h*ode_ex(t)         # and compute the next value


# plt.plot(tpoints,xpoints)    # and create a plot to show the results
# plt.xlabel('time t')
# plt.ylabel('x(t)',rotation=0)
# plt.grid()
# plt.title("Euler's Method")
# plt.show()


#####################
# ######## Code 2: dx(t)/dt is a function in x and t
# import numpy as np                   # import packages
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint            # this is the ODE solver

# #### dx/dt = x
# def ode_ex(x,t):
#     return(x)

# a = 0                        # set up the bounds for the problem; from a...
# b = 2                        # to b...
# N = 1000                     # want 1000 steps
# h = (b-a)/N                  # and this is the step size
# x_init = 1                        # and this is the initial value

# tpoints = np.arange(a,b,h)   # create the array of steps
# xpoints = []                 # the empty list of results

# x = x_init
# for t in tpoints:            # at each of the steps
#     xpoints.append(x)        # put in the current value
#     x += h*ode_ex(x,t)       # and compute the next value




# # # odeint arguments: function which calculates the derivative
# # #                   intial value
# # #                   an array of time steps
# # #                   optional: args=()
# # # and it returns an arra of values

# # x_odeint = odeint(ode_ex, x_init, tpoints)   # call the solver

# #### A funvtion with arguments
# # A_CONST = 1
# # def ode_ex_w_args(x, tpoints, A_CONST):
# #     return x * A_CONST

# # x_odeint_w_args = odeint(ode_ex_w_args, x_init, tpoints, args= (A_CONST,))   # call the solver


# print("exp(2) =", np.exp(2))
# plt.plot(tpoints,xpoints)    # and create a plot to show the results
# # plt.plot(tpoints,x_odeint)    # and create a plot to show the results
# # # plt.plot(tpoints,x_odeint_w_args)    # and create a plot to show the results
# plt.xlabel('time t')
# plt.ylabel('x(t)',rotation=0)
# plt.grid()
# plt.title("Euler's Method")
# plt.show()



#####################
######## Code 3: Runge-Kutta Method with Degree 2:

# #### dx/dt = -x**3 + sin(t)
# #### Euler's method x(t+h) = x(t) + hf(x,t)

# import numpy as np                   # import packages
# import matplotlib.pyplot as plt

# def ode_ex(x,t):
#     return(-x**3 + np.sin(t))

# ###### First: Accurate Method: (Euler with small step size)
# a = 0                               # set up the bounds for the problem; from a...
# b = 10                              # to b...
# N_ACCURATE  = 1000                     # Large N to result in a small step size
# h_small = (b-a)/N_ACCURATE                  
# x_small = 0.0                       # and this is the initial value

# tpoints_small = np.arange(a,b,h_small)   # create the array of steps
# xpoints_small = []                 # the empty list of results

# for t in tpoints_small:            # at each of the steps
#     xpoints_small.append(x_small)        # put in the current value
#     x_small += h_small*ode_ex(x_small,t)       # and compute the next value

# ##### Second Euler Method with large step size (Runge-Kutta method with degree 1)

# N_EULER = 30                       # Large step size
# h_large = (b-a)/N_EULER                  
# x_large = 0.0                      # and this is the initial value

# tpoints_large = np.arange(a,b,h_large)   # create the array of steps
# xpoints_large = []                 # the empty list of results

# for t in tpoints_large:            # at each of the steps
#     xpoints_large.append(x_large)        # put in the current value
#     x_large += h_large*ode_ex(x_large,t)       # and compute the next value


# ##### Third Runge-Kutta Method with larrge step size and degree 2:
# N_RK = 30                            # want 10 steps
# h_RK = (b-a)/N_RK                         # and this is the step size
# x_RK = 0.0                             # and this is the initial value

# tpoints_RK = np.arange(a,b,h_RK)          # create the array of steps
# xpoints_RK = []                        # the empty list of results

# for t in tpoints_RK:                   # at each of the steps
#     xpoints_RK.append(x_RK)               # put in the current value
#     k1 = h_RK*ode_ex(x_RK,t)              # compute the intermediate factors
#     k2 = h_RK*ode_ex(x_RK+0.5*k1,t+0.5*h_RK)
#     x_RK += k2                         # and the new value


# # point_to_examine = 6
# # accurate_x = xpoints_small[np.where(tpoints_small == tpoints_large[point_to_examine])[0].item(0)]
# # print(f'Euler error at (t = {point_to_examine})        =',np.abs(xpoints_large[point_to_examine]-accurate_x))
# # print(f'Runge-Kutta error at (t = {point_to_examine})  =',np.abs(xpoints_RK[point_to_examine]-accurate_x))

# plt.plot(tpoints_small,xpoints_small, label='Accurate')    # and create a plot to show the results
# plt.plot(tpoints_large,xpoints_large, label = 'RK Deg 1 (Euler)')    # and create a plot to show the results
# plt.plot(tpoints_RK,xpoints_RK,label = 'RK Deg 2')           # and create a plot to show the results
# plt.xlabel('time t')
# plt.ylabel('x(t)',rotation=0)
# plt.grid()
# plt.legend()
# plt.title("Comparing Effect of Degree")
# plt.show()


# ######################
# ######### Code 4:

# # infinite time ODE example using 4th-order Runge-Kutta
# # author: olhartin@asu.edu updated by sdm

# import numpy as np                       # import packages
# import matplotlib.pyplot as plt

# # dx/dt = 1/(x**2 + t**2) and solve from t=0 to t=infinity
# # so let t = u/(1-u)

# def ode_inf(x,u):
#     return( 1 / ( ( x**2 * ( 1 - u )**2 ) + u**2 ) )

# a = 0                                 # 0->infinity becomes 0->1
# b = 1
# N = 100                               # do 100 time steps
# h = (b-a)/N                           # the step size

# upoints = np.arange(a,b,h)            # create the array of times
# tpoints = []                          # hold the values converted from u to t
# xpoints = []                          # hold the values of x

# x = 1.0                               # initial value
# for u in upoints:                     # for each of the time steps...
#     tpoints.append(u/(1-u))           # convert back to t
#     xpoints.append(x)                 # add to the value list
#     k1 = h*ode_inf(x,u)               # compute the Runge-Kutta factors
#     k2 = h*ode_inf(x+0.5*k1,u+0.5*h)
#     k3 = h*ode_inf(x+0.5*k2,u+0.5*h)
#     k4 = h*ode_inf(x+k3,u+h)
#     x += (k1+2*k2+2*k3+k4)/6          # get the next value
    
# plt.plot(tpoints,xpoints)             # and plot the result
# plt.xlim(0,80)
# plt.xlabel('time t')
# plt.ylabel('x(t)',rotation=0)
# plt.grid()
# plt.title('Infinite Range Example')
# plt.show()



# ##################
# #########Code 5:
# import numpy as np                            # import packages
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint            # this is the ODE solver

# # solving dy/dt = -2y initial condition y(0)=1

# ################################################################################
# # Function to return the derivative a position with respect to time.           #
# # Inputs:                                                                      #
# #    ypos - current position                                                   #
# #    time - current time                                                       #
# # Outputs:                                                                     #
# #    returns the derivative                                                    #
# ################################################################################

# def calc_derivative(ypos, time):
#     return -2 * ypos

# time_vec = np.linspace(0, 4, 40)              # create the time steps

# # odeint arguments: function which calculates the derivative
# #                   intial value
# #                   an array of time steps
# #                   optional: args=()
# # and it returns an arra of values

# yvec = odeint(calc_derivative, 1, time_vec)   # call the solver


# # a_const = 1
# # def calc_derivative1(ypos, time, a_const):
# #     return -2 * ypos * a_const
# # yvec1 = odeint(calc_derivative1, 1, time_vec, args= (a_const,))   # call the solver

# plt.plot(time_vec,yvec)                       # and plot the results
# # plt.plot(time_vec,yvec1)                       # and plot the results
# plt.xlabel('time')
# plt.ylabel('y',rotation=0)
# plt.title('odeint')
# plt.show()