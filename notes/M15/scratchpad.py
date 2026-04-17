import secrets
import numpy as np

seed = secrets.randbits(128)
# 103848893776354803619115694269950729738
rng = np.random.default_rng(seed)
