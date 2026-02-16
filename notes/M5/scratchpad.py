import pandas as pd
import numpy as np
from sklearn import datasets

data = {'Name': ['Alice', 'Bob', 'Charlie', 'Chris'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'Los Angeles', 'Chicago', 'San Diego']}

df = pd.DataFrame(data)
# print(df.iloc[1, 0])

iris = datasets.load_iris()
# print(iris.feature_names)

print(np.tri(3,3,-1))

print(pd.DataFrame(['Name', 'City']))