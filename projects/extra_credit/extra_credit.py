# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 27 February 2026

# Extra Credit Project

# I did not use AI at all to complete this assignment

#################################
##### Constants and Imports #####
#################################

import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import time
import sys
import os

# Define the number of bits, the number of samples, the frequency of the sinewave, and whether to make a graph of SNR vs BER at the end.
NUM_BITS = 10_000
NUM_SAMP = 20
FREQ = 10e6
PERIOD = 1 / FREQ
OMEGA = 2*np.pi*FREQ
MAKE_GRAPH = False

###########################################
##### Generate the samples with noise #####
###########################################

def generate_data(snr_dB: float):
    """This function generates the data used by both the machine learning algorithm and the classical algorithm.\n

       *****Inputs*****\n
       snr_dB: This is the signal to noise ratio expressed in dB. This is used to determine the noise standard deviation.\n
       
       *****Outputs*****\n
       random_bits: This includes all of the randomly generated bits. We use this as the answer key.\n
       sample_times: The points in time when we sampled the waveform. This is used in the classical demodulation algorithm.\n
       dirty_array: The sample values of the sinewave we created, with the noise added to it."""

    # Find the noise standard deviation associated with this SNR value.
    snr_rat = 10**(snr_dB/10)
    noise_sigma = np.sqrt(0.5 / snr_rat)

    # Create a numpy random number generator object and generate some random bits.
    rng = np.random.default_rng()
    random_bits = rng.integers(0, 2, NUM_BITS)

    # These are the points in time at which the sinewave will be sampled.
    sample_times = np.linspace(0, PERIOD, NUM_SAMP)

    # Fill in the clean array based on those sample points defined above.
    clean_array = np.zeros([NUM_BITS, NUM_SAMP])
    for idx, _ in enumerate(clean_array):
        clean_array[idx] = np.cos(OMEGA*sample_times + random_bits[idx]*np.pi)

    # Generate a noise array and add that to the clean array, producing the dirty array.
    noise_array = rng.normal(0, noise_sigma, [NUM_BITS, NUM_SAMP])
    dirty_array = clean_array + noise_array

    return random_bits, sample_times, dirty_array

############################################
##### Classical demodulation functions #####
############################################

def estimate_phase(sample_times: np.array, sample_values: np.array, freq_guesses: np.array):
    """This function will estimate the phase of a sampled waveform.\n

       ****Inputs*****\n
       sample_times: The points in time when we sampled the waveform. This is used in the classical demodulation algorithm.\n
       sample_values: The sample values of the sinewave we created, with the noise added to it.\n
       freq_guesses: The frequency values that we want to test. The final answer will rely on the correct frequency being in this array.\n

       *****Outputs*****\n
       phase_est: This is the algorithm's estimate for what the phase of that waveform should be."""
    
    J = list()
    h = np.zeros((NUM_SAMP, 2))
    for freq in freq_guesses:
        h[:,0] = np.cos(2*np.pi*freq*sample_times)
        h[:,1] = np.sin(2*np.pi*freq*sample_times)
        a = np.dot(h.T, sample_values)
        b = inv(np.dot(h.T, h))
        c = np.dot(b, a)
        d = np.dot(h, c)
        J.append(np.dot(sample_values.T, d))

    idx_max = np.argmax(J)
    f_est = freq_guesses[idx_max]

    h[:, 0] = np.cos(2*np.pi*f_est*sample_times)
    h[:, 1] = np.sin(2*np.pi*f_est*sample_times)

    a = np.dot(h.T, sample_values)
    b = inv(np.dot(h.T, h))
    c = np.dot(b, a)

    d = np.arctan(abs(c[1]/c[0]))
    phase_est = np.where(c[0] > 0, d, np.pi-d)
    return phase_est

def evaluate_estimate(phase_est: float, correct_bit: float):
    """This function will evaluate whether the received phase allows the receiver to correctly reconstruct the original data.\n
       If the phase is greater than pi/2 or less than -pi/2, then we interpret that to be a bit 1.\n
       We compare that with the correct bit to make our decision.\n
    
       *****Inputs*****\n
       phase_est: The phase estimate for this waveform at the receiver.\n
       correct_bit: The actual bit used to create this waveform at the transmitter.\n

       *****Outputs*****\n
       bool: The function will return True if the received bit matches the transmitted bit, and False otherwise."""

    # If the phase estimate is closer to a phase of -pi, then interpret this bit as a 1. Otherwise 0.
    if (phase_est > np.pi/2) or (phase_est < -np.pi/2):
        bit_est = 1
    else:
        bit_est = 0
    
    # If the bit was interpreted correctly, then return True. Otherwise False.
    if bit_est == correct_bit:
        return True
    else:
        return False

###########################################################
##### Estimate the phase using classical demodulation #####
###########################################################

def demod_cl(random_bits: np.array, sample_times: np.array, dirty_array: np.array):
    """This function demodulates the received signals using the classical methods.\n
    
       *****Inputs*****\n
       random_bits: This includes all of the randomly generated bits. We use this as the answer key.\n
       sample_times: The points in time when we sampled the waveform. This is used in the classical demodulation algorithm.\n
       dirty_array: The sample values of the sinewave we created, with the noise added to it.\n

       *****Outputs*****\n
       ber: The bit error rate using the classical demodulation technique."""

    # Initialize the array containing which frequencies we are going to guess and a variable storing the number of bit errors.
    freq_guesses = np.linspace(FREQ/10, FREQ*10, 20)
    bit_errors = 0

    # Loop over every bit in the generated data and determine whether that bit was interpreted correctly at the receiver.
    for idx, sample_values in enumerate(dirty_array):
        phase_est = estimate_phase(sample_times, sample_values, freq_guesses)

        # Add one to the count of bit errors if the noise caused the receiver to interpet the bit wrong.
        if not evaluate_estimate(phase_est, random_bits[idx]):
            bit_errors += 1

    # Calculate the bit error rate and return that value to the next level.
    ber = bit_errors / random_bits.size
    return ber

#####################################################
##### Estimate the phase using machine learning #####
#####################################################

def demod_ml(random_bits: np.array, dirty_array: np.array, num_neigh=100):
    """This function demodulates the signal using machine learning, more specifically the KNN algorithm.\n
       
       *****Inputs*****
       random_bits: This includes all of the randomly generated bits. We use this as the answer key.\n
       dirty_array: The sample values of the sinewave we created, with the noise added to it.\n
       num_neigh: The number of neighbors used for the KNN algorithm. Default value is set to 100.\n

       *****Outputs*****
       ber: The bit error rate for the machine learning algorithm."""

    # Create the test and training split.
    df = pd.DataFrame(data=dirty_array)
    X_train, X_test, y_train, y_test = train_test_split(df, random_bits, test_size=0.5)

    # Standardize the data using a standard scalar.
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Create the KNN classifier object and train it.
    knn = KNeighborsClassifier(n_neighbors=num_neigh, weights='distance')
    knn.fit(X_train_std, y_train)

    # Use the trained KNN classifier to predict the test outputs, then calculate an accuracy score and bit error rate.
    y_pred = knn.predict(X_test_std)
    acc = accuracy_score(y_test, y_pred)
    ber = 1 - acc
    return ber

#############################
##### Program Execution #####
#############################

if __name__ == "__main__":

    # Generate the data and demodulate using both the machine learning and classical algorithms.
    # Next, print that information to the terminal as described in the prompt.
    random_bits, sample_times, dirty_array = generate_data(snr_dB=-10)

    # cl stands for the classical algorithm.
    # ml stands for the machine learning algorithm, KNN in this case.  
    ber_cl = demod_cl(random_bits, sample_times, dirty_array)
    ber_ml = demod_ml(random_bits, dirty_array)
    print([round(ber_cl, 3)])
    print([round(ber_ml, 3)])
    
    # If the user chooses not to make the graph, then terminate the program here. 
    # I will leave that variable False when I turn the code in. If you want to see the graph, then just make it True.
    if not MAKE_GRAPH:
        sys.exit()

    # Create arrays of the SNR values and bit error rates for each demodulation technique.
    snr_values = np.arange(-20, 10, step=1)
    ber_vals_ml = np.zeros(snr_values.size)
    ber_vals_cl = np.zeros(snr_values.size)

    # Start a timer used to track progress.
    t0 = time.time()

    # Loop over each SNR value in the array, calculating the bit error rates for each one.
    for idx, snr_dB in enumerate(snr_values):

        # Generate the data using the appropriate amount of noise for this SNR value.
        random_bits, sample_times, dirty_array = generate_data(snr_dB)

        # Demodulate the waveform using both the machine learning algorithm and the classical demodulation technique.
        ber_cl = demod_cl(random_bits, sample_times, dirty_array)
        ber_ml = demod_ml(random_bits, dirty_array)

        # Add those results to arrays for each algorithm.
        ber_vals_cl[idx] = ber_cl
        ber_vals_ml[idx] = ber_ml

        # I want updates on the program's progress so that I can make sure the runtime is reasonable.
        t1 = time.time()
        print(f'snr = {snr_dB}; elapsed = {t1-t0:.3f} s')

    # Create a graph comparing the bit error rates for the machine learning and classical techniques.
    plt.scatter(snr_values, ber_vals_cl, label='Classical Method')
    plt.scatter(snr_values, ber_vals_ml, label='Machine Learning')
    plt.xlabel('SNR Values (dB)')
    plt.ylabel('Bit Error Rates (Ratio)')
    plt.title('SNR vs BER for Various Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig('projects/extracredit/ber_vs_snr.png', dpi=300)
    plt.show()

