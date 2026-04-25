import matplotlib.pyplot as plt
import numpy as np

fsig = 1e3
Tsig = 1 / fsig
fsamp = 2.2e3
Tsamp = 1 / fsamp

tsamp = np.arange(0, 10*Tsig, Tsamp)
ysamp = np.sin(2*np.pi*fsig*tsamp)

tsig = np.arange(0, 10*Tsig, Tsamp/50)
ysig = np.sin(2*np.pi*fsig*tsig)

plt.plot(tsig, ysig, color='black')
plt.scatter(tsamp, ysamp, color='red')
plt.title(f'fsamp/fsig = {fsamp/fsig:.1f}')
plt.grid(True)
plt.show()