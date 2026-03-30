# for traveling salesman examples
# author: sdm

# NOTE: If you change N here, be sure to change
# N in the salesman code!

import numpy as np                # import packages
import matplotlib.pyplot as plt
from random import random

N = 100                           # number of cities

# choose N city locations and calculate the initial distance
cities = np.empty([N,2],float)    # 2D array for x and y

cities[:,0] = np.random.rand(N)
cities[:,1] = np.random.rand(N)

my_file = open("m12_cities.txt",'w')
for index in range(N):
    my_file.write(str(cities[index,0])+' '+str(cities[index,1])+'\n')

my_file.write('0.0 0.0\n')

my_file.close()
