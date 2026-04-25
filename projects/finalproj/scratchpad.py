import torch
import time

N = 10_000

# A = torch.randn(N, N)
# B = torch.randn(N, N)

# t0 = time.time()
# C = torch.matmul(A, B)
# print(f'CPU Time: {time.time()-t0:.2f} s')

A = torch.randn(N, N).cuda()
B = torch.randn(N, N).cuda()

t0 = time.time()
C = torch.matmul(A, B)
print(f'GPU Time: {time.time()-t0:.2f} s')