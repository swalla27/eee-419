import numpy as np
import pandas as pd
import re

x = np.ones((12,1))
print(x.shape)
print(x.T.shape)
y = np.dot(x,np.transpose(x))
print(y)