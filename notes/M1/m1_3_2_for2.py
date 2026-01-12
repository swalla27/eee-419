# program to show more for loop functionality
from numpy import arange      # needed to get an array
from numpy import linspace    # needed to show linspace

for index in range(10):       # show how else works
    if ( index > 4 ):
        print(index,"is bigger than 4")
    else:
        print(index,"is less than or equal to 4")
else:
    print("printing this!")

print("done")

input()     # pause here

for index in range(10):       # show how break works
    if ( index > 4 ):
        print(index,"is bigger than 4")
        break                 # stop inner loop!
    else:
        print(index,"is less than or equal to 4")
else:
    print("not printing this!")

print("done")

input()    # pause here

for index in arange(0,10):  # these are ints; still no last value
    print(index)            # arange returns an array!

input()    # pause here

for index in linspace(0,10,11):  # these are now floats; last value included
    print(index)                 # arange returns an array!
