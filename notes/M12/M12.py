# ###### Fourier Transform Basics:
# # example of extracting signal from noise
# # author: allee updated by sdm

# import numpy as np                                      # import packages
# import matplotlib.pyplot as plt
# from scipy import fftpack

# # create a noisy sine wave
# period = 1                                            # seconds
# time_end = 2
# time_vec = np.linspace(0, time_end, 101)                  # create array of times
# time_step = time_vec[1]                           # seconds

# # create the signal
# sig = np.sin( ( 2 * np.pi / period ) * time_vec)  # pristine wave

# plt.plot(time_vec,sig)                        # plot the clean wave
# plt.xlabel('time (s)')
# plt.ylabel('Amplitude')
# plt.title('Time Domain')

# sig_fft = fftpack.fft(sig)                       # execute the fft
# freq_step = 1/time_end
# sample_rate = 1/time_step
# print(type(sample_rate))
# max_freq_content = sample_rate / 2
# freq_vec = np.arange(0,max_freq_content*2+freq_step,freq_step)
# print(len(freq_vec))
# # print(np.imag(sig_fft))

# plt.figure()
# plt.plot(freq_vec,np.real(sig_fft))                # and plot it
# plt.xlabel('frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.title('Frequency Domain')
# plt.show()



# #################################
# ######### FFT Shift:

# import numpy as np                                      # import packages
# import matplotlib.pyplot as plt
# from scipy import fftpack

# # create a noisy sine wave
# period = 0.5                                            # seconds
# time_end = 2
# time_vec = np.linspace(0, time_end, 101)                  # create array of times
# time_step = time_vec[1]                           # seconds

# # create the signal
# sig = np.sin( ( 2 * np.pi / period ) * time_vec)  # pristine wave

# plt.stem(time_vec,sig)                        # plot the clean wave
# plt.xlabel('time (s)')
# plt.ylabel('Amplitude')
# plt.title('Time Domain')

# sig_fft = fftpack.fftshift(fftpack.fft(sig))                       # execute the fft
# freq_step = 1/time_end
# sample_rate = 1/time_step
# print((sample_rate))
# max_freq_content = sample_rate / 2
# freq_vec = np.arange(-max_freq_content,max_freq_content+freq_step,freq_step)

# plt.figure()
# plt.stem(freq_vec,np.real(sig_fft))                # and plot it
# plt.xlabel('frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.title('Frequency Domain')
# plt.show()


# ##############################
# ########## Three sinusoids:
# import numpy as np                                      # import packages
# import matplotlib.pyplot as plt
# from scipy import fftpack

# # create a noisy sine wave
# period1 = 1                                            # seconds
# period2 = 0.5                                            # seconds
# period3 = 0.1                                            # seconds
# time_end = 20
# time_vec = np.linspace(0, time_end, 1001)                  # create array of times
# time_step = time_vec[1]                           # seconds
# print(time_step)

# # create the signal
# sig1 = np.sin( ( 2 * np.pi / period1 ) * time_vec)  # pristine wave
# sig2 = np.sin( ( 2 * np.pi / period2 ) * time_vec)  # pristine wave
# sig3 = np.sin( ( 2 * np.pi / period3 ) * time_vec)  # pristine wave
# sig = sig1 + sig2 + sig3

# # plt.plot(time_vec,sig)                        # plot the clean wave
# # plt.xlabel('time (s)')
# # plt.ylabel('Amplitude')
# # plt.title('Time Domain')

# sig_fft = fftpack.fftshift(fftpack.fft(sig))                       # execute the fft
# freq_step = 1/time_end
# sample_rate = 1 / time_step
# max_freq_content = sample_rate / 2
# freq_vec = np.arange(-max_freq_content,max_freq_content+freq_step,freq_step)

# plt.figure()
# plt.stem(freq_vec,np.real(sig_fft))                # and plot it
# plt.xlabel('frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.title('Frequency Domain')
# plt.show()




# ##############################
# ########## Real World Signals (needs axis to your own audio file)


# from pydub import AudioSegment
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import fftpack
# import sounddevice as sd


# # Load the M4A file (Try a male audio file then a female audio file):
# audio = AudioSegment.from_file("audio.m4a", format="m4a")

# # Get the raw samples as a list (or numpy array if numpy is installed)
# samples = audio.get_array_of_samples()

# sample_rate = audio.frame_rate

# print(sample_rate)

# # create a noisy sine wave
# time_step = 1 /sample_rate                                       # seconds
# period = 1.0                                            # seconds
# # time_vec = np.arange(0, 20, time_step)                  # create array of times

# # create the signal
# sig_clean = samples
# # sig_noise = 0.5 * np.random.randn(time_vec.size)        # some noise
# # sig = sig_clean + sig_noise                             # combine them

# # plt.plot(time_vec,sig_clean,'r')                        # plot the clean wave
# # plt.xlabel('time (s)')
# # plt.ylabel('signal')
# # plt.title('Clean Signal')
# # plt.show()

# # plt.plot(time_vec,sig_noise,'g')                        # plot the clean noise
# # plt.xlabel('time (s)')
# # plt.ylabel('signal')
# # plt.title('Noise')
# # plt.show()

# # signal = signal[0:len(signal)/2]
# sample_rate = 1 / time_step
# sig_dur = len(sig_clean) / sample_rate
# freq_step = 1/sig_dur
# max_freq_content = sample_rate / 2
# freq_vec = np.arange(-max_freq_content,max_freq_content,freq_step)
# sig_clean_fft = fftpack.fftshift(fftpack.fft(sig_clean))
# plt.plot(freq_vec,sig_clean_fft,'b')                              # plot the combination
# plt.xlabel('freq')
# plt.ylabel('signal')
# plt.show()
