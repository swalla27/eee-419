# stock exchange example
# author: allee updated by sdm

import numpy as np                  # import packages
import matplotlib.pyplot as plt
from scipy import fftpack

# read in stock data...
with open('/home/steven-wallace/Documents/asu/eee-419/notes/M12/m12_dow.txt') as f:
    dowdata = []                            # initialize a list
    for line in f:                          # for every line...
        line = line.split()                 # split into individual "words"
        if line:                            # skip blank lines
            line = [float(i) for i in line] # can be than 1 data per line
            dowdata.extend(line)            # line is now a list, so extend
            
dowdata = np.array(dowdata)                 # convert the data into an array
ddata = np.ndarray.flatten(dowdata)         # and make it 1xN instead of Nx1

plt.plot(ddata/1000)                        # plot the data (assume x)
plt.xlabel('sample')                        # scale by 1000
plt.ylabel('signal')
plt.title('DOW Data')
plt.show()

# perform a fast Fourier transform - the r in rfft indicates real data
# (ie not complex)
ddata_fft = np.fft.rfft(ddata)
power = np.abs(ddata_fft)                   # and compute its power

powerdB = np.log10(power)                   # base 10 log to even things out
plt.plot(powerdB)                           # and plot the power
plt.xlabel('frequency')
plt.ylabel('amplitude dB')
plt.title('FFT of DOW Data')
plt.show()

ddata_fft_filt = ddata_fft[:21]             # keep just the first 21 frequencies
reconstruct = np.fft.irfft(ddata_fft_filt)  # and do the inverse fft
plt.plot(reconstruct/1000)                  # and plot the result
plt.xlabel('sample')                        # scale by 1000
plt.ylabel('signal')
plt.title('iFFT of DOW Data keeping top 21 frequencies')
plt.text(2,300,'<- Note the artifact!')
plt.show()

# Repeat with the discrete cosine transform; same comments apply as above...
# Scipy has dct and idct while numpy does not
# Note that the issue at the start disappears with the dct
ddata_dct = fftpack.dct(ddata)

# Since the signal is real, just plot the postive frequencies
power_dct = np.abs(ddata_dct)

powerdB_dct = np.log10(power_dct)           # base 10 log to even things out
plt.plot(powerdB_dct)                       # and plot the power
plt.xlabel('frequency')
plt.ylabel('amplitude dB')
plt.title('DCT of DOW Data')
plt.show()

ddata_dct_filt = ddata_dct[:21]             # keep just the first 21 frequencies
reconstruct_dct = fftpack.idct(ddata_dct_filt) # and do the inverse fft
plt.plot(reconstruct_dct)                   # and plot the result
plt.xlabel('sample')
plt.ylabel('signal')
plt.title('iDCT of DOW Data keeping top 21 frequencies')
plt.show()
