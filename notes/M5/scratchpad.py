import numpy as np

rng = np.random.default_rng()
x = ['cat', 'dog', 'fish']
print(rng.choice(x, 10))

