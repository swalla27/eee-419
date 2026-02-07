import numpy as np
import random
import sys

# Problem 1
N = 1_000_000
bits = np.random.randint(0, 2, N)
print(sum(bits)/len(bits))

# Problem 2
p = 0.7
data = list()
for _ in range(N):
    random_number = random.random()
    if random_number < p:
        data.append(1)
    else:
        data.append(0)
print(sum(data)/len(data))

# Problem 3
options = [1, -1]
data = list()
for _ in range(N):
    data.append(random.choice(options))
print(sum(data)/len(data))

# Problem 4
M = 10

probabilities = list()
for i in range(2, M+1):
    probabilities.append(1/i)

data = list()
for _ in range(N):
    new_value = np.random.choice(np.arange(2, M+1), p=probabilities)
    data.append(new_value)
print(sum(data)/len(data))