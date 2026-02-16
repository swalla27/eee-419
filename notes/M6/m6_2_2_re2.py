# more regular expression examples
import re

# greedy vs nongreedy... determined by another used of ?
str8 = 'here is <abcd/> and here is another <efgh/> and the end'
greedy = re.search('<.*/>',str8)        # get LONGEST string from '<' to '>'
print(greedy.group())
not_greedy = re.search('<.*?/>',str8)   # now get the SHORTEST such string
print(not_greedy.group())
input()                                 # pause here...

# now extract the stuff between the brackets
not_greedy = re.search('<(.*?)/>',str8) # we want what's between the brackets
print(not_greedy.group())
print(not_greedy.group(1))
input()                                 # pause here...

# find something at the end of the string
# $ fixes the target string to the end of the searched string
str9 = 'xxone and yyone and zzone'
print(re.search('..one$',str9).group()) # remember . matches any char here
print(re.findall('..one$',str9))        # see - only one found!
input()                                 # pause here...

# find something at the start of the string
# ^ fixes the target string to the beginning of the searched string
strA = 'onexx and oneyy and onezz'
print(re.search('^one..',strA).group())
print(re.findall('^one..',strA))        # see - only one found!
input()                                 # pause here...

# ignoring case
strB = 'dog Dog doG DOG'
dogs = re.findall('dog',strB,re.IGNORECASE)
print(dogs)

# more complex examples...
match = re.search(r'\d\s*\d\s*\d', 'xx1 2   3xx') # * is zero or more!
print(match.group())

match = re.search(r'\d\s*\d\s*\d', 'xx12  3xx')
print(match.group())

match = re.search(r'\d\s*\d\s*\d', 'xx123xx')
print(match.group())

# Suppose we have a text with many email addresses
# Here re.findall() returns a list of all the found email strings
str = 'purple alice@fakeco.com, blah monkey bob@fakeco.com blah dishwasher'
emails = re.findall(r'[\w.-]+@[\w.-]+', str)
print(emails)
for email in emails:
    # do something with each found email string
    print(email)

# extract the username and host separately...
tuples = re.findall(r'([\w.-]+)@([\w.-]+)', str)
print(tuples)
for tuple in tuples:
    print(tuple[0]) # username
    print(tuple[1]) # host
