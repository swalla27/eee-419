# script to illustrate tuples

my_tup = ( 1, 2 )       # create a tuple
print(my_tup[0])        # get values from the tuple
print(my_tup[-1])
print()                 # create an empty line

for item in my_tup:     # can use for loops with tuples!
    print(item)

print()                 # create an empty line
my_tup[0] = 4           # generates an error!
