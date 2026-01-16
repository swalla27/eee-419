# splitting up a string and file I/O

# split up a string into words and iterate on list entries rather than index

help(str.split)                          # first, see how this works
my_string = "Once upon a time"
print("this is my_string:",my_string)
my_words = my_string.split()             # get a list of words in my_string
print("my_words is a list:",my_words)
print("my_string did not change:",my_string)

for word in my_words:                    # iterate based on items - nice!
    print("iterate on list items:",word)

for index in range(len(my_words)):       # iterate on the index
    print("iterate by index:",my_words[index])

input()     # pause here

# strings have native split operations...
str1 = 'what a fine day this is'        # create a string
print(str1.split())                     # default split is white space
print(str1.split('f'))                  # split on 'f'
print(str1.split('i'))                  # split on 'i'
input()                                 # pause here...

# can join strings from lists or tuples...
delim = '@'                             # use this as the joining character
my_tuple = ('a', 'b', 'c')              # an example tuple
my_list  = ['d', 'e', 'f']              # an example list
print(delim.join(my_tuple))             # put them together...
print(delim.join(my_list))              # put them together...
input()                                 # pause here...

delim = ''                              # don't like anything in between?
print(delim.join(my_tuple))             # put them together...
print(delim.join(my_list))              # put them together...
input()                                 # pause here...

delim = ' '                             # put a space between
print(delim.join(my_tuple))             # put them together...
print(delim.join(my_list))              # put them together...
input()                                 # pause here...

# open a file for reading or writing

print("notice that there is no file in the directory called m2_str_io.out")
input()     # pause to verify

my_inp_file = open("m2_3_1_str_io.txt","r")      # r is default, add a b for binary
my_out_file = open("m2_3_1_str_io.out","w")      # to append, use "a" instead of "w"

for line in my_inp_file:          # neat way to get a line at a time to process
    print(line)                   # note the extra carriage returns
    my_out_file.write(line)       # write it out

my_inp_file.close()               # always close when done
my_out_file.close()               # forgetting to close is a common error

print("now the file is there...")
input()     # pause here

my_inp_file = open("m2_3_1_str_io.txt","r")    # r is default, add a b for binary

one_line = my_inp_file.readline()      # read one line at a time
print(one_line)

another_line = my_inp_file.readline()
print(another_line)                    # carriage return from file is printed!
print("end of print")
remove_stuff = another_line.strip()    # strip gets rid of it
print(remove_stuff)                    # there is also lstrip and rstrip
print("end of print")

del_d = remove_stuff.strip('d')        # strip out the 'd'
print(del_d)                           # the 'd' is gone!
my_inp_file.close()
