# more dictionary stuff...
import csv                     # needed to create comma separated output

# adding more entries - order not always maintained
# don't access via index - access via key!

my_dict = {'hat':'fedora','shirt':'polo','pants':'jeans','socks':'argyle'}
print(my_dict)
item_list = list(my_dict.items())
value_list = list(my_dict.values())
key_list = list(my_dict.keys())
print("item list:",item_list)
print("value list:",value_list)
print("key list:",key_list)

input()     # pause here

# use a dictionary to create a histogram...
# if there's already a dictionary entry for a character, increment it
# if not, start it at 0!

mystring = "a funny thing happened on the way to the forum"
char_dict = {}

for char in mystring:
    print(char)
    char_dict[char] = char_dict.get(char,0) + 1

print(char_dict)

input()      # pause here

# make a list of dictionaries and sort them

keys = ['Name','Age','Year']         # define the dictionary keys
students = []                        # initialize list so we can append to it
filename = 'notes/M2/m2_4_2_dict2.txt'            # string with the filename
fh = open(filename,"r")              # open the file

lines = fh.readlines()               # create a list of the lines in the file
fh.close()                           # always close the file!
print(lines)                         # check them out

for line in lines:                   # for all the lines we got
    line=line.strip()                # strip out carrage control line feed
    studnt = line.split()            # split out data on default delimiter blank
    student = dict(zip(keys,studnt)) # "zips" up keys with corresonding values
    students.append(student)

print("original student list:")
print(students,"\n")                  # what have we got?

# NOTE: we'll learn about lambda in module 4...
# for now, just understand that it is taking the place of a function.

# this is how to sort a list of dictionaries
students.sort(key=lambda d: (d['Name']))
print("sorted student list (by name):\n",students,"\n")

# this is how to sort a list of dictionaries
students.sort(key=lambda d: (d['Age']))
print("sorted student list (by age):\n",students,"\n")

# this is how to sort a list of dictionaries
students.sort(key=lambda d: (d['Year']))
print("sorted student list (by year):\n",students,"\n")

input()      # pause here

# how to create a comma separated list for a spreadsheet...
# open a file and assign it to csvfile

with open('notes/M2/m2_4_2_dict2.csv','w',newline='') as csvfile:
    # establish a writer with the keys
    writer = csv.DictWriter(csvfile, fieldnames=keys)
    writer.writeheader()                # comma separated list of keys

    for student in students:            # now for each dictionary in the list...
        writer.writerow(student)        #    comma separated list of values
