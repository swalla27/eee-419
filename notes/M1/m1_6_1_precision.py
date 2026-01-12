# examples for controlling output precision
# we've seen lots of print statements... how to round a float?

my_float = 34.829
print("round to 2 past the decimal:",round(my_float,2))    # hundredths!
print("round to 1 past the decimal:",round(my_float,1))    # tenths!
print("round to 0 past the decimal:",round(my_float,0))    # ones!
print("round to 1 ante the decimal:",round(my_float,-1))   # tens!

input()     # pause here

# another method

val_0 = 2.334567e20                           # number with an exponent
print('round w/ exp:\t{:.3e}'.format(val_0))  # round with exponent

big_num = 123456789.23                        # too many digits!
print("with commas:\t{:,}" .format(big_num))  # just commas

val_1 = 123456.12387                          # combine the methods
print('combined:\t{:,.3f}'.format(val_1))     # with commas but float

input()     # pause here

# print in hex - note that we precede the hex number with 0x so people looking
# at it realize it is a hex number and not something else! And note that we use '+'
# rather than ',' to separate 0x from the value so we do NOT get a space!

val_2 = 29                                    # create an integer
print("decimal\t",val_2)                      # print in decimal
print("hex\t 0x"    + '{:x}' .format(val_2) ) # print as a hex - for programmers!
print("hex\t 0x"    + '{:X}' .format(val_2) ) # or in capital letters
print("octal\t 0o"  + '{:o}' .format(val_2) ) # prints octal
print("binary\t 0b" + '{:b}' .format(val_2) ) # prints binary

