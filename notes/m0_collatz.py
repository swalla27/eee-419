#####################################################
# Program to implement the Collatz Conjecture:      #
# for any positive integer:                         #
#    if even, divide by 2                           #
#    if odd, multiply by 3 and add 1                #
#    continue until running result is 1             #
# conjecture (unproven!) -> this always leads to 1! #
#####################################################

# request a number from the user as the starting point:
value = int(input("enter a number: "))

loop_cnt = 0                       # track the number of iterations required
while value != 1 :                 # requires valid conjecture!
    loop_cnt += 1                  # track iterations

    if ( value % 2 ) == 0:         # if even...
        value //= 2                #    NOTE: using integer divide!
    else:
        value = ( value * 3 ) + 1  # NOTE - this always yields an even number!

    print("   ", value)            # print the current value

print(loop_cnt, "iterations")      # Done! Print the iterations
