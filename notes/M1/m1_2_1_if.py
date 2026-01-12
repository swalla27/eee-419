# Program to illustrate 'if-elif-else'

my_val = int(input("Give me a number: "))    # input always returns a string!!!

if ( my_val > 0 ):                           # NOTE - this line ends with a colon
    print("Got a good value!")               #    Indent by 4 spaces
    if ( my_val < 10 ):
        print("It's kind of small:",my_val)  # indent another 4 spaces
    elif ( my_val < 100 ):                   # back out 4 spaces to end if clause!
        print("It's medium in size:",my_val) # again, indent 4 spaces
    elif ( my_val < 1000 ):                  # back out to end prior clause
        print("It's a big one:",my_val)      # in again
    else:                                    # need the colon here too
        print("It's a whopper:",my_val)

print("all done")
