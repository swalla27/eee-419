import numpy as np
import pandas as pd
import re

x = '### Subject: scuba diving at... From: steve.millman@asu.edu Body: Underwater, where it is at'
match = re.search(r'[\w]*at$', x)
if match:
    print(match.group())
else:
    print('no match')