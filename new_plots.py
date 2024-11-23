import os
import wave
import contextlib
import matplotlib.pyplot as plt
import glob
import soundfile as sf
import numpy as np
from utils_com import *


"""This code was designed by us, but have used ChatGPT for optimization, troubleshooting, and plots."""

# =========================
# AudioMNIST Data
# =========================

# Base directory containing subfolders
base_directory = 'original audio/AMNISTdata/'

# List to store durations of all .wav files
durations = []

# Traverse through all subfolders 01 to 60
for i in range(1, 61):  # 1 to 60 inclusive
    subfolder = os.path.join(base_directory, f'{i:02}')  # 01, 02, ...
    wav_files = glob.glob(os.path.join(subfolder, '*.wav'))  # Get all .wav files in the subfolder

    for file_path in wav_files:
        with contextlib.closing(wave.open(file_path, 'r')) as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            durations.append(duration)

# Create a histogram for AudioMNIST
plt.figure(figsize=(10, 6))
plt.hist(durations, bins=20, edgecolor='black')
plt.title('AudioMNIST Data Duration Distribution', fontsize=15)
plt.xlabel('Duration (seconds)', fontsize=15)
plt.ylabel('Number of Files', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(False)
#plt.show()

# =========================
# Urbansound Data
# =========================

# Base directory containing subfolders
base_directory = 'original audio/Urbansound/'

# List to store durations of all .wav files
durations = []

# Traverse through all subfolders 01 to 10
for i in range(1, 11):  # 1 to 10 inclusive
    subfolder = os.path.join(base_directory, f'fold{i:02}')  # fold01, fold02, ...
    wav_files = glob.glob(os.path.join(subfolder, '*.wav'))  # Get all .wav files in the subfolder

    for file_path in wav_files:
        # Read file properties using soundfile
        data, samplerate = sf.read(file_path)
        duration = len(data) / float(samplerate)
        durations.append(duration)

# Create a histogram for Urbansound
plt.figure(figsize=(10, 6))
plt.hist(durations, bins=20, edgecolor='black')
plt.title('Urbansound8K Data Duration Distribution', fontsize=15)
plt.xlabel('Duration (seconds)', fontsize=15)
plt.ylabel('Number of Files', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(False)
#plt.show()



file_path = "original audio/AMNISTdata/06/0_06_6.wav"
y, sr = librosa.load(file_path, sr = None)
padded_y = pad_segment(y, 47998)
spec = stft(padded_y, frame_size=512, overlap=256)
epsilon = 1e-4
spec_db = np.array(20 * np.log10(np.abs(spec) + epsilon))

# Plot the original padded signal with consistent x-axis range
# Time axis for the original signal
time = np.linspace(0, len(padded_y)/sr, len(padded_y))

# Time axis for the spectrograms
n_frames_spec = spec_db.shape[1]  # Number of time frames in your custom spectrogram
#n_frames_lib = lib_db.shape[1]  # Number of time frames in the librosa spectrogram

time_spec = np.linspace(0, len(padded_y)/sr, n_frames_spec)
#time_lib = np.linspace(0, len(padded_y)/sr, n_frames_lib)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle(f'Original Signal (0_06_6.wav) and Spectrogram', fontsize=15)

# Plot the original padded signal
ax[0].plot(time, padded_y)
ax[0].set_xlim([0, len(padded_y)/sr])
ax[0].set_ylabel("Amplitude")
ax[0].label_outer()

# Plot your custom spectrogram with proper time axis
img1 = ax[1].imshow(spec_db, aspect='auto', origin='lower', extent=[0, len(padded_y)/sr, 0, sr//2], cmap='magma', interpolation='none')
ax[1].set_xlim([0, len(padded_y)/sr])
ax[1].set_ylabel("Frequency (Hz)")
ax[1].label_outer()
ax[1].set_xlabel("Time (s)")


# Adjust layout
plt.tight_layout()
plt.show()