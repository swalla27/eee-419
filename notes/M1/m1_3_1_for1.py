# examples of range and for loops
from numpy import arange           # needed to create float range
from numpy import linspace         # needed to create linear space range
from numpy import full             # needed to create arrays

my_rng = range(10)            # this is a type: it only stores start/stop value!
print("The range is:",my_rng) # so this just prints it as is without evaluating!
print("Make it a list:",list(my_rng),"\n")  # force the iterator

for index in range(10):            # count 0 to 9
    print(index)

input()    # pause here

for index in range(0,10,2):        # count 0 to 9 by 2
    print(index)

input()    # pause here

for index in range(10,0,-1):       # count 10 to 1
    print(index)
else:
    print("finished the loop")

input()    # pause here

my_nums = arange(10)               # arange returns an array!
print("The arange is:",my_nums)    # we learn about arrays in a future lecture
x = list(my_rng) - my_nums
y = my_rng - my_nums
print("x =",x)
print("y =",y)
print("type of list(my_nums):",type(list(my_rng)))
print("type of my_rng:",type(my_rng))
print("type of my_nums:",type(my_nums))
# z = 0 - my_rng
input()

for index in arange(0,1,.1):       # these are now floats; still no last value
    print(index)                   # note strange values!

input()    # pause here

# now linspace examples

my_linsapce = linspace(0,1,10)              # now get the ends!
print("linspace is:",my_linsapce)
print("\n")

my_linsapce_noend = linspace(0,1,10,False)  # now do not get the ends!
print("linspace without end is:",my_linsapce_noend)
print("\n")

my_linspace_float = linspace(0,10,11)       
print("linspace floats:",my_linspace_float)
print("\n")

my_linspace_int,step = linspace(0,10,10,retstep=True,dtype=int)
print("linspace step was",step)
print(my_linspace_int)              # note the missing number!
print("\n")

my_linspace_complex = linspace(0,5+5j,6,dtype=complex)
print("linspace complex:",my_linspace_complex)
print("\n")

a = full(3,1)     # create two arrays
b = full(3,5)     # details on this in a later lecture...
print(a,b,"\n\n")

new_arr = linspace(a,b,5,axis=0)    # horizontal stepping
print(new_arr,"\n\n")

new_arr = linspace(a,b,3,axis=1)    # vertical stepping
print(new_arr)

input()    # pause here

# example of linspace with a for loop
print("linspace in a for loop")
for val in linspace(0,2,5):
    print(val)

input()    # pause here

# enumerate is used to get the index with a value
print('enumerating')
alphs = ['a','b','c']
for index,value in enumerate(alphs):
    print(index,value)
