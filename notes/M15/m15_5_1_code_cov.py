# Example code for use with code coverage

################################################################################
# A function with conditions...                                                #
# Inputs: in1, in2, in3                                                        #
# Outputs: out1, out2                                                          #
################################################################################

def func(in1,in2,in3):
    if in1 > 0 :
        print(in2)
    else:
        print(in3)

    if ( in1 < 10 ) and ( in2 > 10 ) :
        out1, out2 = in1, in2
    else:
        out1, out2 = in3, in3

    return out1, out2

# start of main code

int1 = int(input("input an integer: "))
int2 = int(input("input an integer: "))
int3 = int(input("input an integer: "))

result1, result2 = func(int1,int2,int3)
print(result1,result2)
