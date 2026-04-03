# Create frequency chart for musical instruments
# author: allee updated by sdm

import numpy as np                            # import packages
import matplotlib.pyplot as plt
from scipy import fftpack

instruments = ["violin", "piano", "trumpet"]  # the instruments

# read in musical instrument
# Note that we split line so that it becomes a list. And then we are able to
# apply the int() function to all the elements of the list in one go.
# This is another way to map a function onto a list.
for instr in instruments:

    with open('/home/steven-wallace/Documents/asu/eee-419/notes/M12/m12_'+instr+'.txt') as f:      # open the file
        musicdata = []                        # create an empty list
        for line in f:                        # for every line in the file
            line = line.split()               # deal with blanks and multiples
            if line:                          # blank lines are skipped...
                line = [int(i) for i in line] # can be more than 1 data per line
                musicdata.extend(line)        # line is now a list, so extend

    mdata = np.array(musicdata)               # turn the list into an array

    plt.plot(mdata)                           # and plot the data
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.title(instr)
    plt.show()

    # perform a Fast Fourier Transform @ typical value for recordings
    time_step = 1/44100                       # 44100 samples per second
    sample_freq = fftpack.fftfreq(mdata.size, d=time_step)  # set up fftpack
    mdata_fft = fftpack.fft(mdata)                          # and transform

    # Since the signal is real, just plot the postive frequencies
    pidxs = np.nonzero(sample_freq > 0)             # indices where freq > 0
    freqs = sample_freq[pidxs]                      # select those frequencies
    power = np.square(np.abs(mdata_fft)[pidxs])     # compute their power
    plt.plot(freqs[1:1000],power[1:1000])           # plot power vs freq
    plt.xlabel('frequency (Hz)')
    plt.ylabel('amplitude')
    plt.title(instr)
    plt.show()
