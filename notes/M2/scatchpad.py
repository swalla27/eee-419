import pandas as pd

s = pd.Series(list('abcde'), index=[95, 96, 97, 0, 1])
print(s.iloc[1])


