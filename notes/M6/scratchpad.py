import re

str4 = "from: 12389087 f453.589 subject: nothing of interest"

print(re.findall(r'\d{3}',str4))