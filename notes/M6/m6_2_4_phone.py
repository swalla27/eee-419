# example of directives that don't consume charaters
# author: sdm (problem suggested by E Graves)

import re                                             # the package

numbers_1 = '321 456-0011 first (480) 123-4567 '      # assemble the string
numbers_2 = 'second 480-987-6543 third 4561234567 '
numbers_3 = 'fourth 999213.5678 fifth 222-2323 '
numbers_4 = 'sixth 333.5432 seventh 480618-1111'
numbers = numbers_1 + numbers_2 + numbers_3 + numbers_4
print(numbers)
print()

# match (word boundary)(area code without parens)(optional separator)
area_code_1     = r'\b(?:\d{3}[ -.]?)?'

# match (word boundary)(area code with parens)(optional separator)
# NOTE: word boundary won't work since ( is not a word character!
# so use ^ for start of string or (?<=\s) for preceding white space
area_code_2     = r'(?:^|(?<=\s))(?:\(\d{3}\)[ -.]?)?'

# match (3-digit exchange)(optional separator)
three_dig_space = r'\d{3}[ -.]?'

# match (4-digit number)(word boundary)
four_dig        = r'\d{4}\b'

# put together a phone number with no parens around the area code
full_1          = area_code_1 + three_dig_space + four_dig

# put together a phone number with parens around the area code
full_2          = area_code_2 + three_dig_space + four_dig

# now combine them with the OR operator
find_num        = full_1 + r'|' + full_2

# find the numbers and print then one at a time!
phone_numbers = re.findall( find_num, numbers)
for num in phone_numbers:
    print(num)
print()

