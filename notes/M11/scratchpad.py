import numpy as np
import matplotlib.pyplot as plt

N = 100_000
rng = np.random.default_rng()

SNR = 5
SIGMA = np.sqrt(1 / (2*SNR))

x = rng.choice([-1, 1], size=N)
y = x + rng.normal(0, SIGMA, size=N)
y_hat = np.where(y > 0, 1, -1)

indices = (x != y_hat)
err_vals = x[indices]
errors = len(err_vals)
print(errors/N)