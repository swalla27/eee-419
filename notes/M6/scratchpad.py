import re

str4 = "from: 1238 f453.589 subject: nothing of interest"

print(re.findall(r'\b(\d{3,})',str4))