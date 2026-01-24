from numpy import arange           # needed to create float range
# from numpy import linspace         # needed to create linear space range
# from numpy import full             # needed to create arrays
# from numpy import array
from numpy import pi
import numpy as np

print(pi)


# # Two main categories: Lists and Arrays
# my_list = [0,1.0,'cat',False]
# print((my_list))
# x=np.array(my_list)
# print(type(x)) #[1]
my_rng = range(3)
# print("my_rng is:",my_rng)
# print("list(my_rng) is:",list(my_rng))    # Force the iterator, get a list
# print("type of my_rng:",type(my_rng))
# print("type of list(my_rng):",type(list(my_rng)))
# print("===================")
my_nums = arange(10)               # arange returns an array!
# print("my_nums is:",my_nums)    # we learn about arrays in a future lecture
# print("type of my_nums:",type(my_nums))
# print("list(my_nums) is:",list(my_nums))
# Can we add a number to a range? array? list?
# z = list(my_rng) + 5
z = list(my_rng) + [5]
# print(z)
z1 = [1, 4, 6]
z.extend(z1)
z.append(z1)   # compare
# print(z)
z = my_nums + arange(1,11)
print(z) #[1  2  3  4  5  6  7  8  9 10]
# z = list(my_nums) + 1
# # Can we add array to range? to list?
# x = my_rng + my_nums
# print("x =",x)
# y = list(my_rng) + my_nums
# print("y =",y)


################
my_linspace_int , step= np.linspace(0,10,10,retstep=True)
print(my_linspace_int)
print("linspace step was",step)
# note: step is 1.111…12, not 1.111…11
# print("\n")

# my_linspace_int , step = linspace(0,10,10,retstep=True)
# print(my_linspace_int)              # note the missing number!
# print("linspace step was",step)
# print("\n")

# Force linspace to give only integers
my_linspace_int,step = np.linspace(0,10,10,retstep=True,dtype=int)
print(my_linspace_int)              # note the missing number!
print("linspace step was",step)
print("\n")

######################
# Adding lists
num_list = [1 , 2 , 'cat' , 'silver']
print(num_list)
num_list_b = [5 , 4 , 3 , 2]
x = num_list + num_list_b
print("x=",x)
num_list.append(num_list_b)
print("After appending num_list=",num_list)
num_list.extend(num_list_b)
print(num_list)
num_list.append([5])
num_list.append(5)
print(num_list)
# # Remember: “append an item to a list” and “extend a list to a list”


print("====================")
new_list = num_list
print("new_list is",new_list)
new_list[1] = 8
print(new_list,", Its address is",hex(id(new_list)))
print(num_list,", Its address is",hex(id(num_list)))

# # here's how to copy a list...
num_list = [0, 1, 2, 3]
cp_list = num_list.copy()
cp_list[1] = 37
print(num_list,hex(id(num_list)))
print(cp_list,hex(id(cp_list))) #different memory address than num_list


# # mapping onto a list
# num_list = [1 , 2 , 3]
# map_float_list = map(float,num_list)
# float_list = list(map(float,num_list)) # Force iterator
# print(map_float_list,hex(id(map_float_list)))
# print(float_list,hex(id(float_list)))


# #############################
# my_list = [1,2,3]
# print(my_list,hex(id(my_list)))
# my_list.clear()
# print(my_list,hex(id(my_list))) # a cleared list still exists (i.e. its pointer exists)


# #############################
# Unpacking entries in a list
# Sometimes needed when calling functions!
my_list1 = [1,2,3]
my_list2 = [4,5,6]
list3 = [ my_list1, 99, my_list2 ]
list4 = [ *my_list1, 99, *my_list2 ] # asterisk unpacks the list
print("sublists:",list3)
print("all unpacked:",list4)
print(hex(id(my_list1)),*my_list1) # Unlike in C, these two commands aren't equivalent


#################
a = np.full(3,1)     # create two arrays
b = np.full(3,5)     # details on this in a later lecture...
c = range(3)
print(a,b,"\n\n")

new_arr = np.linspace(a,b,5,axis=0)    # horizontal stepping
print(new_arr,"\n\n")

new_arr = np.linspace(a,b,5,axis=1)    # vertical stepping
print(new_arr, "\n\n")

new_arr = np.linspace(c,b,5,axis=0)    # horizontal stepping
print(new_arr)





## Ungraded Training Problems:
# solve them if you are rusty on programming or if you are a begginner in programming.
# No need to submit this anywhere, not even in the HW submission)
# Problem 1:
# Quadratic equation solver + identify whether the roots are complex, real, equal

# Problem 2:
# Implement the factorial function using recursion
# Then again using for loop

# Problem 3:
# Write a code that implements the exponential function (e**x) using Taylor's expansion (google what that is)