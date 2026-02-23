import re           # bring in the package
import sys

# define a sample string

my_str = 'Cinderella met a fella at the ball. They all played on the XS57B but not Cinderella, who played on the S23X.'

# find the longest string starting and ending with ll
long_lls = re.search(r'll.*ll',my_str)
print(long_lls.group())

# find the shortest strings starting and ending with ll
long_lls = re.findall(r'll.*?ll',my_str)
print(long_lls)

# find all occurances of Cind and the rest of the word it starts at the START of the string
cind = re.findall(r'^Cind\w*',my_str)
print(cind)

# extract the model number that Cinderella did NOT play on... hint: it must start with XS and end with B
didnot = re.findall(r'(?<=XS)\d+(?=B)',my_str)
print(didnot)

# extract all serial numbers... hint: they start and end with 1 or more uppercase letters with numbers between
allnums = re.findall(r'[A-Z]+\d+[A-Z]+',my_str)
print(allnums)

# IDs start with X or Y and end with _####
# find the names of the people who have IDs in this sentence
my_str2 = 'First id is Xsally_2137 and the second is Yomar_5389'
names = re.findall(r'(?<=[XY])\w+(?=_\d{4})', my_str2)
print(names)