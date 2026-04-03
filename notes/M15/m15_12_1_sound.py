# example that plays sound
# author: cwang135; fade added by and comments added by sdm

import numpy as np                          # packages needed
import matplotlib.pyplot as plt
import sounddevice as sd                    # package to play sound

# define music note frequencies used in the song in Hz
fg3 = 196;                                  # g3 is G in the 3rd octave
fc4 = 261.63;                               # c4 is C in the 4th octave
fg4 = 392;                                  # g4 is G in the 4th octave

fs = 44100       # sampling rate/frequency: # of samples per second (Hz)
one_beat = 0.5;  # time duration of a 1/4 note in sec

# generate time arrays for 1/3 of a beat and for 2 beats
# note that the time step is 1/fs as defined above
t_third = np.arange(0, one_beat/3, 1/fs)
t_two = np.arange(0, one_beat*2, 1/fs)

# create note sine waves                  
g3_third = np.sin(2 * np.pi * fg3 * t_third);
c4_two = np.sin(2 * np.pi * fc4 * t_two);
g4_two = np.sin(2 * np.pi * fg4 * t_two);

# if you don't cross fade, you get a click when notes start/stop
# this is a simple linear fade in/out; more complex fades are typically used
fade_len = 100                          # apply the fade for this many samples
start_mask = np.linspace(0,1,fade_len)  # fade for start of a note
end_mask   = np.linspace(1,0,fade_len)  # fade for the end of a note

g3_third[:fade_len] *= start_mask       # apply the fades
g3_third[-fade_len:] *= end_mask
c4_two[:fade_len] *= start_mask
c4_two[-fade_len:] *= end_mask
g4_two[:fade_len] *= start_mask
g4_two[-fade_len:] *= end_mask

# plot entire arrays (uncomment the next four lines if needed)
#plt.plot(t_third, g3_third, 'g-', label='g3_third')
#plt.plot(t_two, c4_two, 'b--', label='c4_two')
#plt.plot(t_two, g4_two, 'r:', label='g4_two')
#plt.show()

# plot five hundred values in the arrays for reference
start_index = 500
end_index = 1000
plt.plot(t_third[start_index:end_index],
         g3_third[start_index:end_index],
         'g-', label='g3_third')
plt.plot(t_two[start_index:end_index],
         c4_two[start_index:end_index],
         'b--', label='c4_two')
plt.plot(t_two[start_index:end_index],
         g4_two[start_index:end_index],
         'r:', label='g4_two')

plt.xlabel('time')
plt.ylabel('value')
plt.title("three notes")
plt.legend()
plt.show()

# assemble notes into a song
song = np.concatenate([g3_third, g3_third, g3_third, c4_two, g4_two])
# plot the song - you'll need to make the plot wide to see anything
# t = np.arange(0, one_beat*5, 1/fs) # total five beats in song
# plt.plot(t, song)
# plt.show()

# send audio to play
# Normalize to 0 ~ 2^15-1
audio = song * (2**15 - 1) / np.max(np.abs(song))

# Convert to 16-bit data
audio = audio.astype(np.int16)

# Start playback
sd.play(audio, fs)

# Wait until file is done playing
status = sd.wait()  
