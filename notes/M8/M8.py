# # Script 1:
# import numpy as np                       # for arrays
# from scipy import optimize               # access to optimization functions
# import matplotlib.pyplot as plt          # to create the plot
# from mpl_toolkits.mplot3d import Axes3D  # to allow 3D!

# ################################################################################
# # Function to implement the quadratic function to which we find the minimum    #
# # Input:                                                                       #
# #    coords - x value at which to evaluate the function                        #
# # Output:                                                                      #
# #    the value of the function                                                 #
# ################################################################################

# def quadd(coords):
#     x = coords             # extract the x and y for clarity
#     return ( (x-2)**2 -4)

# x = np.linspace(-4, 4)      # allow x to vary from -4 to 4
# print("x= ",x)

# plt.figure()                   # simple visualization in 2D
# plt.plot(x,quadd(x))
# plt.xlabel('x')
# plt.title('Quadratic Function in 1D')


# # First optimization using bfgs with initial guess 0, disp=0 for quiet
# result = optimize.fmin_bfgs(quadd, 10000, disp=1)
# print('bfgs result:',result)

# print()
# # Second optimization using basin hopping with initial guess 0
# result = optimize.basinhopping(quadd, 10000)
# print(type(result))                        # this is a significant structure
# print('BH: Optimal Solution =',result.x)    # Returns the optimal solution
# print('BH: Optimal Value =',result.fun)    # Returns the optimal value
# print('BH: Termination Reason:',result.message)    # Returns the optimal value
# plt.show()



# Script 2 (Stuck in local minimum, perturbation needed):

import numpy as np                       # for arrays
from scipy import optimize               # access to optimization functions
import matplotlib.pyplot as plt          # to create the plot
from mpl_toolkits.mplot3d import Axes3D  # to allow 3D!

################################################################################
# A single-variable function to with 2 local minima and 1 global minimum       #
# Input:                                                                       #
#    coords - x value at which to evaluate the function                        #
# Output:                                                                      #
#    the value of the function                                                 #
################################################################################

def fun_w_loc_min(x):
    y = -0.75
    return ( 4 - ( 2.1 * x**2 ) + ( x**4 / 3. ) ) * x**2 +  \
           x * y +                                          \
           (-4 + ( 4 * y**2 ) ) * y**2

x = np.linspace(-2, 2)      # allow x to vary from -4 to 4
print("x= ",x)

# First optimization using bfgs with initial guess 200, disp=0 for quiet
result = optimize.fmin_bfgs(fun_w_loc_min, 2, disp=0)
print('bfgs result:',result)

print()
# Second optimization using basin hopping with initial guess 200
result = optimize.basinhopping(fun_w_loc_min, 200)
print('BH: Optimal Solution =',result.x)    # Returns the optimal solution
print('BH: Optimal Value =',result.fun)    # Returns the optimal value
print('BH: Termination Reason:',result.message)    # Returns the optimal value

plt.figure()                   # simple visualization in 2D
plt.plot(x,fun_w_loc_min(x))
plt.xlabel('x')
plt.title('Quadratic Function in 1D')
plt.show()



# # Script 3 (Two variables):

# # Example from scipy.org for nonlinear optimization - find the minimum
# # from Allee, updated by sdm

# import numpy as np                       # for arrays
# from scipy import optimize               # access to optimization functions
# import matplotlib.pyplot as plt          # to create the plot
# from mpl_toolkits.mplot3d import Axes3D  # to allow 3D!

# ################################################################################
# # Function to implement the 6-hump shape from which to find the minimum        #
# # Input:                                                                       #
# #    coords - x and y value at which to evaluate the function                  #
# # Output:                                                                      #
# #    the value of the function                                                 #
# ################################################################################

# def sixhump(coords):
#     x = coords[0]             # extract the x and y for clarity
#     y = coords[1]
#     return ( 4 - ( 2.1 * x**2 ) + ( x**4 / 3. ) ) * x**2 +  \
#            x * y +                                          \
#            (-4 + ( 4 * y**2 ) ) * y**2

# # create a meshgrid so we have two arrays from which we can project the surface
# x = np.linspace(-2, 2)      # allow x to vary from -2 to 2
# y = np.linspace(-1, 1)      # allow y to vary from -1 to 1
# xg, yg = np.meshgrid(x, y)  # create the meshgrid

# # plt.figure()                   # simple visualization in 2D
# # plt.imshow(sixhump([xg, yg]))
# # plt.colorbar()
# # plt.xlabel('x')
# # plt.ylabel('y',rotation=0)
# # plt.title('six hump in 2D')
# # plt.show()

# # First optimization using bfgs with initial guess 2,-1, disp=0 for quiet
# result = optimize.fmin_bfgs(sixhump, [2,1], disp=0)
# print('bfgs result:',result)

# # Second optimization using basin hopping with initial guess 2,-1
# result = optimize.basinhopping(sixhump, [2,1])
# print(type(result))                        # this is a significant structure
# print('basin hopping result:',result.x)    # many other attributes generated

# fig = plt.figure()                         # awesome visulization in 3D
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(xg, yg, sixhump([xg, yg]), rstride=1, cstride=1,
#                        cmap=plt.cm.jet, linewidth=0, antialiased=False)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.zaxis.set_rotate_label(False)
# ax.set_zlabel('f(x, y)')
# ax.set_title('Six-hump Camelback function')
# plt.show()






# Script 4: Non-Linear Equations
# # ######## First: After obtaining a single equation

# import numpy as np
# from scipy.optimize import fsolve

# # Define the system of equations
# def x2(x1):
#     x2 = 2 - x1
#     return x2


# def f2(x1):
#     # f2 = x1**2 + (2-x1)**2 - 4    # Second equation
#     f2 = x1**2 + (x2(x1))**2 - 4    # Second equation
#     return f2

# # Initial guess
# initial_guess = 500000   # some initial guesses might not converge, try others!

# # Solve the system using fsolve
# x1_sol = fsolve(f2, initial_guess)

# # Display the solution
# print("Solution for x1 =",x1_sol, ", x2 =", x2(x1_sol))
# print("f2(x1) =", f2(x1_sol))


# ######## Second: If you cannot obtain a single equation: 

# import numpy as np
# from scipy.optimize import fsolve

# # Define the system of equations
# def equations(vars):
#     x1, x2 = vars  # Unpack variables
#     eq1 = x1 + x2 - 2          # First equation
#     eq2 = x1**2 + x2**2 - 4    # Second equation
#     return [eq1, eq2]

# # Initial guess
# initial_guess = [1, 1]

# # Solve the system using fsolve
# solution = fsolve(equations, initial_guess)

# # Display the solution
# print("Solution for x1 and y2:", solution)
# print("f1(x1, x2) =", equations(solution)[0])
# print("f2(x1, x2) =", equations(solution)[1])
 


# # Script 5 (Least Squares Arbitrary Function):

# # nonlinear least squares example
# # perhaps similar to david gay dn2fb
# # author: allee@asu.edu updated by sdm

# import numpy as np                    # math and arrays
# from scipy.optimize import leastsq    # optimization function
# import matplotlib.pyplot as plt       # so we can plot

# N=5000                                # number of samples to generate

# ################################################################################
# # Function to fit                                                              #
# # Inputs:                                                                      #
# #    kd - first value, which we are trying to determine                        #
# #    p0 - second value, which is the known domain                              #
# # Output:                                                                      #
# #    value of the function                                                     #
# ################################################################################

# def func(kd,p0):
#     return 0.5*(-1-((2*p0)/kd) + np.sqrt(4*(p0/kd)+(((2*p0)/kd)-1)**2))

# ################################################################################
# # Function to compute the difference between the actual and predicted values   #
# # Inputs:                                                                      #
# #    kd_guess - the guess for the value of the first parameter                 #
# #    p0 - the second parameter, which is known                                 #
# #    actual - the sampled value                                                #
# # Output:                                                                      #
# #    difference between the actual value and the value calculated with guess   #
# ################################################################################

# def residuals(kd_guess,p0,actual):
#     return actual - func(kd_guess,p0)

# # Create a noisy signal based on a known value of kd of 3
# # since random returns uniformly distributed [0,1), subtract .5 gives
# # [-.5,.5) so we get +/- noise
# kd=3.0                                                # the "known" value
# p0 = np.linspace(0,10,N)                              # an array for p0
# clean = func(kd,p0)                                   # the clean signal
# actual = clean+(np.random.random(N)-0.5)*1.0          # the noisy signal

# # now try to extract the the known value of kd by minimizing the residuals
# # residuals - the function we are optimizing
# # 5 - the initial guess for the value to find, in this case, kd
# # args - the additional arguments needed, in this case the x and y values
# # full_output - return all the outputs
# kd_match,cov,infodict,mesg,ier = leastsq(residuals,1000,args=(p0,actual),
#                                          full_output=True)

# print("actual kd was",kd)                      # original value
# print('kd guess', kd_match)                    # this is the guess for kd
# # print('cov\n',cov)                           # inversion of the Hession
# # print('infodict\n',infodict)                 # various other outputs
# print('mesg\n',mesg)                           # a string with status
# print('ier\n',ier)                             # ier is An integer flag.
#                                                # If it is equal to 1, 2, 3 or 4, the solution was found.
#                                                # Otherwise, the solution was not found. In either case, the optional
#                                                # output variable 'mesg' gives more information






# ############ Script 6: System of linear equations
# ###### Part 1: One Solution
# import numpy as np

# # Define the matrix A and vector b
# A = np.array([[1, 1], [1, -1]])
# b = np.array([[2], [0]])

# print("A =\n", A)
# print("b =\n", b)

# # Compute the solution using A^(-1) * b
# x_inverse = np.dot(np.linalg.inv(A), b)
# # x_pseudo_inverse = np.dot(np.linalg.pinv(A), b)

# print("x =", x_inverse)

# ###### Part 2: Multiple Solutions

# import numpy as np

# # Define the matrix A and vector b
# A = np.array([[1, 1],
#               [2, 2]])
# b = np.array([[2], [4]])

# # Compute the pseudo-inverse solution
# x_pseudo_inverse = np.dot(np.linalg.pinv(A), b)

# print("\nPseudo-inverse solution: x =", x_pseudo_inverse)
# print()


# # ###### Part 3: No Solutions
# import numpy as np

# # Define the matrix A and vector b
# A = np.array([[1, 1],
#               [1, 1]])
# b = np.array([[2], [0]])

# # Compute the pseudo-inverse solution
# x_pseudo_inverse = np.dot(np.linalg.pinv(A), b)

# print("\nPseudo-inverse solution: x =", x_pseudo_inverse)
# print()


# # # ############## Script 7: Finding the parameters of a noisy equation (using Least Squares):

# import numpy as np                    # math and arrays
# from scipy.optimize import leastsq    # optimization function
# import matplotlib.pyplot as plt       # so we can plot

# N=1000                                # number of samples to generate

# ################################################################################
# # Function to fit                                                              #
# # Inputs:                                                                      #
# #    kd - first value, which we are trying to determine                        #
# #    p0 - second value, which is the known domain                              #
# # Output:                                                                      #
# #    value of the function                                                     #
# ################################################################################

# def func(kd,p0):
#     return kd*p0**2+2*p0-5

# ################################################################################
# # Function to compute the difference between the actual and predicted values   #
# # Inputs:                                                                      #
# #    kd_guess - the guess for the value of the first parameter                 #
# #    p0 - the second parameter, which is known                                 #
# #    actual - the sampled value                                                #
# # Output:                                                                      #
# #    difference between the actual value and the value calculated with guess   #
# ################################################################################

# def residuals(kd_guess,p0,actual):
#     return actual - func(kd_guess,p0)

# # Create a noisy signal based on a known value of kd of 3
# # since np.random.random returns uniformly distributed [0,1), subtract .5 gives
# # [-.5,.5) so we get +/- noise
# kd=3.0                                                # the "known" value
# p0 = np.linspace(0,10,N)                              # an array for p0
# clean = func(kd,p0)                                   # the clean signal
# actual = clean+(np.random.random(N)-0.5)*1000          # the noisy signal
# plt.plot(p0,actual)
# plt.show()
# # now try to extract the known value of kd by minimizing the residuals
# # residuals - the function we are optimizing
# # 5 - the initial guess for the value to find, in this case, kd
# # args - the additional arguments needed, in this case the x and y values
# # full_output - return all the outputs
# kd_match,cov,infodict,mesg,ier = leastsq(residuals,5,args=(p0,actual),
#                                          full_output=True)

# print("actual kd was",kd)                      # original value
# print('kd guess', kd_match)                    # this is the guess for kd
# # print('cov\n',cov)                           # inversion of the Hession
# # print('infodict\n',infodict)                 # various other outputs
# print('mesg\n',mesg)                           # a string with status
# print('ier\n',ier)                             # status flag


