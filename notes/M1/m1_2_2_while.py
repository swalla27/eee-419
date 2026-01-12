# Program to illustrate while loops

counter = 8                          # need to set up the tested value

while ( counter >= 0 ):              # We'll repeat as long as the condition is True
    print("count is now:",counter)   # do some stuff...
    counter -= 1                     # make sure you update!

print("Blast off!")

input()   # pause here...

# now, use True and False...
flag = False                          # initialize BEFORE the loop
while ( not flag ):                   # want to loop until the flag is True
    a_val = int(input("Enter 0 to leave: "))
    if ( a_val == 0 ):
        flag = True
    else:
        print("No luck yet...")

print("Finally got a 0!")

input()   # pause here...

# now, use else
flag = True                           # initialize BEFORE the loop
while ( flag ):                       # want to loop until the flag is False
    a_val = int(input("Enter 0 to leave: "))
    if ( a_val == 0 ):
        flag = False
    else:
        print("No luck yet...")
else:                                 # here when loop terminates normally
    print("Finally got a 0!")

input()   # pause here...

# now, use break
flag = False                          # initialize BEFORE the loop
while ( not flag ):                   # want to loop until the flag is True
    a_val = int(input("Enter 0 to leave: "))
    if ( a_val == 0 ):
        break                         # leave the innermost enclosing loop
    else:
        print("No luck yet...")
else:                                 # here when loop terminates normally
    print("don't print this!")

print("print this instead")

input()   # pause here...

# nested loops
x = True                                 # set conditions
y = True
while x:                                 # start the outer loop
    while y:                             # start the inner loop
        print("in inner loop")
        break                            # only breaks out of inner loop
    print("in outer loop")
    x = False                            # gracefully exit output loop
else:
    print("x =", x,"and y =", y)

