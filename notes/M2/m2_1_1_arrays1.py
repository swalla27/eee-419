# Program to illustrate arrays
# 2-D arrays are [row,column]
# Can have arrays of arrays!
# once created, size is fixed (unlike lists!)
# all entries must be of same type (unlike lists!)
# are treated differently than lists!

import numpy as np      # get the array capabilities of numpy

# easy way to create arrays is specify the size with a convenience function
array_0 = np.zeros(4,float)       # need to specify the type of value
array_1 = np.ones(4,complex)      # any type will do
array_2 = np.full([2,3],23)       # can specify an initial value for all entries
array_3 = np.full_like(array_2,2) # make new array same as other but new values
                                  # zeros_like and ones_like work the same way
print(array_0)
print(array_1)
print(array_2)
print(array_3)

input()    # pause here

# make an array directly
mk_array = np.array([11,12,13],int)      # provide a list as an argument
print("make array",mk_array)
my_list = [22,23,24]
my_list_array = np.array(my_list,int)    # provide a list variable
print("list array",my_list_array)

input()    # pause here

# careful of empty and empty_like - they will be filled with whatever
# happened to be in those memory locations. This will look like random garbage!

array_4 = np.empty(3,float)
print(array_4)

input()    # pause here

# now, the fun begins! Python loves arrays!
array_4 = np.zeros(4,int)
print(array_4)

array_4 += 2        # add 2 to all the entries
print(array_4)

array_4 *= 3        # multiply all the entries by 3
print(array_4)

array_5 = np.ones(4,int)
print(array_5)
array_4 += array_5
print(array_4)

input()    # pause here

array_bin_0 = np.full(4,0)
array_bin_1 = np.full(4,1)
array_bin_0[2] = 1
array_bin_0[3] = 1
array_bin_1[0] = 0
array_bin_1[2] = 0
print("",array_bin_0,"\n",array_bin_1)            # note empty quotes to create a space!
array_bin_2 = array_bin_0 & array_bin_1
array_bin_3 = array_bin_0 | array_bin_1
print("and:\n",array_bin_2,"\nor:\n",array_bin_3)

input()    # pause here

# copy works like in lists, so be careful
array_6 = array_5               # points to the same memory!
print(array_5)
array_6[0] = 1000
print("changed",array_5)

array_7 = np.copy(array_5)
array_7[0] = 23
print("\nno change",array_5)
print(array_7)

input()   # pause here
# now, some fun with two arrays
array_x = np.arange( 0,10,1)
array_y = np.arange(10,20,1)
for x_val, y_val in zip(array_x,array_y):  # makes tuples of pairs!
    print(x_val,y_val)

input()   # pause here
# now more fun - use meshgrid to creae a 2D structure
grid_x, grid_y = np.meshgrid(array_x,array_y)
print(array_x,"\n\n")
print(array_y,"\n\n")
print(grid_x,"\n\n")
print(grid_y,"\n\n")
