import re

str4 = "from: steve.millman@asu.edu subject: nothing of interest" 
print(re.search(r'([\w.]+)@([\w.]+)', str4).group(2))