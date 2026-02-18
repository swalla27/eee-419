# interesting string and regular expression examples
import re                               # get the regular expression module
# https://www.w3schools.com/python/python_regex.asp

# using the re module is more powerful as a splitter
# general form: re.split(delimiters,string)
str1 = 'what a fine day this is'        # create a string
print(re.split('[ie]',str1))            # anything between brackets...
print(re.split('i|e',str1))             # or use the | as OR
print(re.split('[a-f]',str1))           # can do ranges!
print(re.split('[\s]',str1))            # can still do white space!
input()                                 # pause here...

# now let's get powerful with repeated characters...
print(re.split('[a-f\s]+',str1))        # + says if it is found 1 or more times
input()

# now let's find things in strings...
str2 = 'at the risk of repeating things I will be repeating things'

finder = re.search('risk',str2)         # look for 'risk' in the string
print("object",finder)                  # finder is a match object!
print("value",finder.group())
print("start",finder.start())
print("end",finder.end())
print("span",finder.span())
print("span start",finder.span()[0])
print("span end",finder.span()[1])
input()                                 # pause here...

finder = re.findall('repeat',str2)      # find all occurrences
for finds in finder:                    # look for all of them...
    print(finds)                        # found both!
input()                                 # pause here...

# now let's start doing pattern matching!
# general form is re.search(pattern,string)
str3 = 'find siiiilly typos here'
finder = re.search(r'iiii',str3)        # find the substring
print(finder)
finder = re.search(r'iixii',str3)       # won't find this...
print(finder)                           # so we get None!
if finder:                              # and None is False!
    print("Got one!")
else:
    print("Nothing there...")
input()                                 # pause here...

# now get words...
str4 = "from: steve.millman@asu.edu subject: nothing of interest"
who = re.search(r'[\w]@[\w]',str4)      # find word char, @, word char
print(who.group())

# allow multiple (1 or more) word chars before and after
who = re.search(r'[\w]+@[\w]+',str4)
print(who.group())

# include . (really a period, not anything!)
who = re.search(r'[\w.]+@[\w.]+',str4)
print(who.group())
input()                                 # pause here...

# try with a dash...
str5 = "from: steve-millman@asu.edu subject: nothing of interest"
who = re.search(r'[\w-]+@[\w.]+',str5)  # dash at the end is just a dash!
print(who.group())
input()                                 # pause here...

# what about things you don't want?
exclude = re.search(r'[^fr\s]+',str4)   # ^ says anything NOT in this set
print(exclude.group())
input()                                 # pause here...

# what if you want something from inside a match?
who = re.search(r'([\w.]+)@([\w.]+)',str4)  # parens say only return this stuff
print(who)                              # all the stuff for the overall match
print(who.group())                      # print the overall match...
print(who.group(0))                     # print the overall match...
print(who.group(1))                     # print the first () match
print(who.group(2))                     # print the second () match
input()                                 # pause here...

# what about trying to match a single character from a set?
# Note that ?: is a special character pair right after a (. It says
# NOT to return what comes inside the parens as () would normally do

# we can do the same sort of thing with [] without needing (?:
# Notice the ? after the "nose" meaning there may be 0 or 1 dashes
# Build a regular expression for emoticons
str6 = 'here is happy :-) and here is sad :( and here is sarcastic :-P'
pat = re.search('[:;=][-]?[\(\)DP]',str6)         # parens ignored inside []
print(pat)
pats = re.findall('[:;=][-]?[\(\)DP]',str6)       # find them all
print(pats)
input()                                           # pause here...

# can do substitutions!
# re.sub(pattern, replacement, str)
str7 = 'this string has a typoo'                  # bad text
new_str = re.sub('[o]+','o',str7)                 # fix it!
print('it was:',str7,"\nbut now it's:",new_str)
