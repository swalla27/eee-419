# example for finding quotes within quotes
# author: M Spears updated by sdm

import re                        # need the package
print()                          # create vertical space

# Here is a string with a quote surrounding other quotes
quote = 'this is "start of stuff "now more stuff" and even "more stuff" final things" and some other stuff'

# first, extract the inner quote using a greedy search
first = re.search('"(.*)"',quote)
print(first.group(1))
print()

# now, extract the inner quotes
second = re.findall('".*?"',first.group(1))
print(second)
print()

# but the above includes the quotation marks - get rid of them!
better = re.findall('"(.*?)"',first.group(1))
print(better)
print()
