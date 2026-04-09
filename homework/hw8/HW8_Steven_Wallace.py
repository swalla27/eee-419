# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 9 April 2026

# Homework 8

# I did not use AI at all to complete this assignment.

from pydub import AudioSegment
import numpy as np
import sounddevice as sd

from scipy import fftpack
import matplotlib.pyplot as plt
import os

# Define constants for the program related to the noise and cutoff frequency.
# I will set both of those Boolean variables to False for the submission.
NOISE_SCALE = 0.1
FREQ_CUTOFF = 2e3
PLAY_AUDIO = False
SHOW_GRAPHS = False

# Load the audio file.
cwd = os.getcwd()
filepath = os.path.join(cwd, 'homework/hw8/Audio_Steven_Wallace.mp3')
audio = AudioSegment.from_file(filepath, format='mp3')

# Get the raw samples.
sig_clean = np.array(audio.get_array_of_samples())
sig_clean = sig_clean / max(sig_clean)

# My file does have two channels, so we are taking odd samples of channel 1.
if audio.channels == 2:
    sig_clean = sig_clean[::2]

# Fetch the sample rate and calculate the time step.
# Sample rate is 44.1 kHz and time step is 22.7 us.
sample_rate = audio.frame_rate
time_step = 1 / sample_rate

# Send the clean signal to the system speaker.
if PLAY_AUDIO:
    sd.play(sig_clean, sample_rate)
    sd.wait()

# Generate random noise and add it to the clean signal.
# The clean signal has 104_467 elements with a max value of 12_232 and min value of -11_080.
rng = np.random.default_rng()
sig_dirty = sig_clean + rng.uniform(low=0, high=NOISE_SCALE, size=len(sig_clean))

# Send the dirty signal to the system speaker.
if PLAY_AUDIO:
    sd.play(sig_dirty, sample_rate)
    sd.wait()

# Find the x and y axes for the frequency domain.
sample_freq = fftpack.fftfreq(len(sig_dirty), d=time_step)
sig_fft = fftpack.fft(sig_dirty)

# I am finding the index of the frequency nearest the cutoff.
x = np.abs(sample_freq - FREQ_CUTOFF)
cutoff_idx = np.argmin(x)

# I will keep all the positive values until the cutoff frequency, and all of its corresponding negative values.
positive_cutoff = cutoff_idx
negative_start = len(sig_fft) - cutoff_idx

# I am making a copy of the original fft signal and then filtering it.
# sigf_lpf means the signal in frequency domain, after a low pass filter.
sigf_lpf = sig_fft.copy()
sigf_lpf[positive_cutoff:negative_start] = 0

# Plot the signal before and after filtering in the frequency domain.
if SHOW_GRAPHS:
    plt.plot(sample_freq, sig_fft, label='Dirty Unfiltered')
    plt.plot(sample_freq, sigf_lpf, label='Dirty Filtered')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectral Intensity')
    plt.title('Signal Comparison, Freq Domain')
    plt.legend()
    plt.grid(True)
    plt.show()

# Perform the inverse Fourier Transform on the filtered signal, taking only the real part.
sigt_lpf = np.real(fftpack.ifft(sigf_lpf))

# Calculate the mean squared error and print it to the terminal.
mse = np.square(sigt_lpf - sig_dirty).mean()
print(f'Mean Squared Error: {mse:.3e}')

# Send the filtered signal to the system speaker.
if PLAY_AUDIO:
    sd.play(sigt_lpf, sample_rate)
    sd.wait()

# Make a graph in the time domain to compare the signal before and after filtering.
if SHOW_GRAPHS:

    # Create an array of the time points for each sample.
    ending_time = time_step * len(sig_clean)
    time_array = np.arange(0, ending_time, step=time_step)

    # Plot the signal before and after filtering in the time domain.
    plt.plot(time_array, sig_dirty, label='Dirty Unfiltered')
    plt.plot(time_array, sigt_lpf, label='Dirty Filtered')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Signal Comparison, Time Domain')
    plt.legend()
    plt.grid(True)
    plt.show()