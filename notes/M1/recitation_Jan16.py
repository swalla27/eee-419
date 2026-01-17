import numpy as np
import math
import time

def factor(x: int):
    factors = set()
    loop_end = math.ceil(np.sqrt(x))+1
    for i in range(1, loop_end):
        if x%i == 0:
            factors.add(i)
            factors.add(x/i)
    return sorted(factors)

def gcd(a: int, b: int):
    while b:
        a, b = b, a%b
    return a

my_nums = np.array([10, 25, 97])
z = my_nums + np.arange(2, 5)

##########################################################################
##### Both of these techniques work for finding the size of an array #####
##########################################################################
print(np.size(z))
print(z.size)

arr1 = np.array([[1, 2, 3]]) # This array has 1 row and 3 columns
print(arr1)
print(arr1.size)
print(arr1.shape)

arr2 = np.array([[1], [2], [3]]) # Whereas this one has 3 rows and 1 column
print(arr2)
print(arr2.size)
print(arr2.shape)

my_linspace, step = np.linspace(0, 10, 11, retstep = True, dtype = int) # You can also make linspace return a step size
print(my_linspace)
print('linspace step was:', step)
print(my_linspace.shape)

a = [1, 2, 5, 10]
b = a
b[1] = 'rat'
print(a) # This is called assignment, meaning both lists point to the same place in memory. Usually this isn't what we want
print(f'address of a: {hex(id(a))}; address of b: {hex(id(b))}') # Same memory addresses

c = [6, 1, 9]
d = c.copy()
c[0] = 'santa claus'
print(d) # This is a list copy operation, which makes them point to different places in memory. This is probably what you want
print(f'address of c: {hex(id(c))}; address of d: {hex(id(d))}') # Different memory addresses

e = [a, b, c]
f = [*a, *b, *c] # You can use an asterisk before a list variable to unpack all of its elements
print(f'packed list: {e}')
print(f'unpacked list: {f}')