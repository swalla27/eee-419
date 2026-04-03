# do FFT on sun spot data
# author: allee updated by sdm

import numpy as np                        # import packages
import matplotlib.pyplot as plt
from scipy import fftpack

# read in sunspot data

with open('/home/steven-wallace/Documents/asu/eee-419/notes/M12/m12_sunspots.txt') as f:       # open the file
    spots = []                            # creae empty lists
    months = []

    for line in f:                        # for every line in the file
        line = line.split()               # blank lines and multiple per line
        if line:                          # blank lines are skipped...
            line = [float(i) for i in line] # convert both strings to numbers
            months.append(line[0])        # the number of the month is first
            spots.append(line[1])         # the number of spots is second

spots = np.array(spots)                   # turn the list into an array
spots = np.ndarray.flatten(spots)         # but need 1xN, so flatten

plt.plot(spots)                           # and plot the data
plt.xlabel('time')
plt.ylabel('signal')
plt.title('sunspot data')
plt.show()

# perform a Fast Fourier Transform
time_step = 1                             # this is the default value
sample_freq = fftpack.fftfreq(spots.size, d=time_step)  # set up fftpack
spots_fft = fftpack.fft(spots)                          # and transform the data

# Since the signal is real, just plot the postive frequencies
pidxs = np.where(sample_freq > 0)               # indices where freq is positive
freqs = sample_freq[pidxs]                      # select those frequencies
power = np.square(np.abs(spots_fft)[pidxs])     # compute their power
plt.plot(freqs[1:1000],power[1:1000])           # plot power vs freq
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude')
plt.title('power vs frequency')
plt.show()

# now try to isolate the peak frequency...
indexmax = np.argmax(power)          # this is the index of the biggest power
print(indexmax)                      # print the index
print(power[indexmax])               # and its power
main_freq = sample_freq[pidxs[0][indexmax]] # get the corresponding frequency
print("frequency:",main_freq)        # and print it
print("period in years:",(1/main_freq)/12)
delta = main_freq * 0.1              # set a guard band for which freqs to keep

# Let's filter out the noise above the main frequency
spots_fft[np.abs(sample_freq) > main_freq + delta] = 0
#spots_fft[np.abs(sample_freq) < main_freq - delta] = 0 # Need to keep low freq!

power = np.abs(spots_fft)[pidxs]          # power of freqs that are left
plt.plot(freqs,power)
plt.xlabel('frequency (Hz)')
plt.ylabel('filtered amplitude')
plt.title('power of filtered frequencies')
plt.show()

filt_sig = fftpack.ifft(spots_fft)        # back to the original space
plt.plot(months,filt_sig)                 # and plot
plt.xlabel('time')
plt.ylabel('filtered signal')
plt.title('filtered sunspot activity')
plt.show()

plt.plot(months[0:700],filt_sig[0:700])   # zoom in a just a few 11-year cycles
plt.xlabel('time')
plt.ylabel('filtered signal')
plt.title('zoom in on 700 months')
plt.show()

