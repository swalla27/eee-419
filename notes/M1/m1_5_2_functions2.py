# illustrate passing lists to functions

################################################################################
# Function mod_list modifies a list                                            #
# inputs:                                                                      #
#   in_list:   the list received                                               #
# outputs:                                                                     #
#   out_list:  the list changed                                                #
################################################################################

def mod_list(in_list):
    in_list[0] = 9                           # change an entry
    in_list.append(6)                        # add an entry
    in_list.pop(1)                           # remove an entry
    
    in_list = ['a','b','c']                  # it's local now! AVOID DOING THIS
    print("in function mod_list:",in_list)   # prove it has changed
    return in_list                           # return the new thing

top_list = [1,2,3,4,5]                       # create a list
print("original list:",top_list)             # show it
got_list = mod_list(top_list)                # call the function
print("new list:",top_list)                  # the list changed
print("returned",got_list)                   # and the returned list

