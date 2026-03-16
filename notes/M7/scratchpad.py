import numpy as np
import pandas as pd
import re

y = "You turned her against me. You have done that yourself."
z = re.search(r'\w+t', y)

print(z.group())