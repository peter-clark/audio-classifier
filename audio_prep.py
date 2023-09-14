import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

#waveform
#fft  --> spectrum
#stft --> spectrogram
#MFCCs

file = "/Users/peterclark/Downloads/Music/19 Awex - It's Our Future (Rock&Roll Mix).mp3"
signal, samplerate = librosa.load(file, sr=22050)
# DISPLAY SIGNAL WAVEFORM
'''
librosa.display.waveshow(signal, sr=samplerate)
plt.xlabel("time")
plt.ylabel("freq")
plt.show()
'''

# FFT
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, samplerate, len(magnitude)) # gives evenly space intervals and presence of data in them
l_frequency = frequency[:int(len(frequency)/2)]
l_magnitude = magnitude[:int(len(frequency)/2)]

# DISPLAY FFT
'''
plt.plot(l_frequency, l_magnitude)
plt.xlabel("freq")
plt.ylabel("magnitude")
plt.show() #still shows with time rather than frequency
'''

# STFT
num_fft = 2048
hop_length = 512
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=num_fft)
spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)

# DISPLAY STFT
'''
librosa.display.specshow(log_spectrogram, sr=samplerate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Freq")
plt.show()
'''

# MFCC
n_mfcc = 13
mfccs = librosa.feature.mfcc(signal, n_fft=num_fft, hop_length=hop_length, n_mfcc=n_mfcc)

# DISPLAY MFCCs
librosa.display.specshow(mfccs, sr=samplerate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Freq")
plt.colorbar()
plt.show()