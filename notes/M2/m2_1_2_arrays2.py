# Program to continue array examples

import numpy as np             # again, the array stuff is in numpy
import os

array_0 = np.zeros(4,int)      # let's get two arrays and put values in
array_1 = np.zeros(4,int)

for index in range(4):         # populate the arrays
    array_0[index] = index
    array_1[index] = ( index * 2 ) + 1

print(array_0)
print(array_1)

array_3 = array_0 * array_1    # element by element multiplication
print(array_3)

input()     # pause here...

# matrix operations are plenty...
# here's how you dot multiply two matrices

array_4 = np.zeros([1,4],int)    # one row, four columns
array_5 = np.zeros([4,1],int)    # four rows, one column
for index in range(4):           # fill the arrays...
    array_4[0,index] = 1 + index
    array_5[index,0] = ( index + 1 ) ** 2

print(array_4)
print(array_5)

my_dot = np.dot(array_4,array_5) # form dot product
print(my_dot)

input()      # pause here

# make sure your sizes are right!
my_dot = np.dot(array_5,array_4) # isn't the same!
print(my_dot)

input()      # pause here

# get the size (total number of elements) and shape (dimensions) of the matrix:
print("array_4 size",array_4.size)       # size gets number of entries
print("array_4 shape",array_4.shape)     # shape gets dimensions
print("array_4 len",len(array_4))        # length gets number of rows!
print("my_dot size",my_dot.size)
print("my_dot shape",my_dot.shape)
print("my_dot len",len(my_dot))          # not expected! (but works for 1-D arrays)

input()      # pause here

# load a matrix from a file...

input_data = np.loadtxt(os.path.join(os.getcwd(), 'notes/M2', "m2_1_2_arrays2.txt"),int)
print(input_data)

input()      # pause here

# slicing of matrices
print(input_data[0,:])
print(input_data[0,2:3])    # remember - end of range, 3, is NOT included
print(input_data[:,2])
print(input_data[:1,3]) # don't include row 1 because that is at the end of the range

