# example of extracting signal from noise
# author: allee updated by sdm

import numpy as np                                      # import packages
import matplotlib.pyplot as plt
from scipy import fftpack

# create a noisy sine wave
time_step = 0.02                                        # seconds
period = 1.0                                            # seconds
time_vec = np.arange(0, 20, time_step)                  # create array of times

# create the signal
sig_clean = np.sin( ( 2 * np.pi / period ) * time_vec)  # pristine wave
sig_noise = 0.5 * np.random.randn(time_vec.size)        # some noise
sig = sig_clean + sig_noise                             # combine them

plt.plot(time_vec,sig_clean,'r')                        # plot the clean wave
plt.xlabel('time (s)')
plt.ylabel('signal')
plt.title('Clean Signal')
plt.show()

plt.plot(time_vec,sig_noise,'g')                        # plot the clean noise
plt.xlabel('time (s)')
plt.ylabel('signal')
plt.title('Noise')
plt.show()

plt.plot(time_vec,sig,'b')                              # plot the combination
plt.xlabel('time (s)')
plt.ylabel('signal')
plt.title('Signal Plus Noise')
plt.show()

# perform a fast Fourier transform on the combined signal
# first, get the DFT sample frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)
sig_fft = fftpack.fft(sig)                             # execute the fft

# Since the signal is real, just plot the postive frequencies
# np.nonzero finds indices of sample_freq meeting the condition
# that the value there is > 0
pidxs = np.nonzero(sample_freq > 0)  # pidxs is a tuple
freqs = sample_freq[pidxs]           # this creates an array with those entries
power = np.abs(sig_fft)[pidxs]       # compute the power of the freqs > 0
plt.plot(freqs,power)                # and plot it
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude')
plt.title('Power at each Frequency')
plt.show()

# find the max freq value
print(power[0:30])                   # print these so can follow the analysis
indexmax = np.argmax(power)          # this is the index of the biggest power
print(indexmax)                      # print the index
print(power[indexmax])               # and its power
use_index = pidxs[0][indexmax]       # translate back to sample_freq index
main_freq = sample_freq[use_index]   # get the corresponding frequency
print(main_freq)                     # and print it
delta = 0.01                         # set a guard band for which freqs to keep

# Let's filter out the noise above or below a given frequency
# Any value in sig_fft whose corresponding value in sample_freq is
# above or below the specified limits is set to 0
sig_fft[np.abs(sample_freq) > main_freq + delta] = 0
sig_fft[np.abs(sample_freq) < main_freq - delta] = 0

# calculate the power of the freqs that are left and plot them
power = np.abs(sig_fft)[pidxs]
plt.plot(freqs,power)
plt.xlabel('frequency (Hz)')
plt.ylabel('filtered amplitude')
plt.title('Filtered Frequencies')
plt.show()

# Perform inverse Fourier transform to go back to the original space
filt_sig = fftpack.ifft(sig_fft)
plt.plot(time_vec,filt_sig)           # and plot
plt.xlabel('time (s)')
plt.ylabel('filtered signal')
plt.title('Cleaned Up Signal')
plt.show()
