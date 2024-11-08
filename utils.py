import numpy as np
import os
import librosa
from scipy.io import wavfile


# def load_data_mnist(folder_path):
#     # Get a list of all files in the folder_path
#     all_files = os.listdir(folder_path)

#     # Determine samplerate of signal
#     sr , _ = wavfile.read(folder_path + "/" + all_files[0])

#     audio_data = []
#     # Load the audio files
#     for file in all_files:
#         if file.endswith('.wav'):
#             y = librosa.load(folder_path + "/" + file, sr=sr)[0]
#             audio_data.append((y,file))

#     return audio_data

def load_data(folder_path):
    # Get a list of all files in the folder_path
    audio_data = []
    for dir, leaves, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                y, sr = librosa.load(dir + "/" + file, sr=None)
                if sr == 44100 or sr == 48000:
                    audio_data.append((y, sr, file))
                    continue
    return audio_data

def trim_signals(data, max_db=20):
    trimmed_audio = []
    for y, file in data:
        if len(y) > 8000:
            continue # Skip audio files longer than 8000 samples
        y_trimmed, _ = librosa.effects.trim(y, ref=np.mean , top_db=max_db)
        trimmed_audio.append((y_trimmed, file))
    return trimmed_audio

def pad_segment(s, window_size):
    dif_sample = abs(len(s) - window_size) # Calculate the differnce in desired signal length and the current signal length
    if len(s) % 2 != 0:
        padded_y = np.pad(s, (dif_sample//2, dif_sample//2 + 1), 'constant', constant_values=(0, 0))
    else:
        padded_y = np.pad(s, (dif_sample//2, dif_sample//2), 'constant', constant_values=(0, 0))
    return padded_y

def stft(x, frame_size=256, overlap=128):
    num_segments = len(x) // overlap - 1 # Calculate the numbxer of segments
    freq_bins = frame_size // 2 + 1 # Define the number of frequency bins
    spec = np.zeros((freq_bins, num_segments)).astype(np.complex128)
    t = 0
    for i in range(0, len(x)-frame_size, overlap):
        seg = x[i:i+frame_size]
        seg = np.hamming(len(seg)) * seg # Apply the hamming window
        if len(seg) < frame_size: # if the segment is shorter than the window size, we need to pad it (usually the last segment)
            seg = pad_segment(seg, frame_size)
        spec[:,t] = np.fft.rfft(seg) 
        t += 1
    return spec


# def gen_spectgrams_mnist(audio_data, padding_length, n_fft=128):
#     spects = []
#     for audio in audio_data:
#         if len(audio[0]) > 8000:
#             continue # Skip audio files longer than 8000 samples
#         y_trimmed, _ = librosa.effects.trim(audio[0], ref=np.mean , top_db=10)
#         padded_y = pad_segment(y_trimmed, padding_length)
#         spec = stft(padded_y, frame_size=n_fft, overlap=n_fft//2)
#         epsilon = 1e-4
#         spec_db = np.array(20 * np.log10(np.abs(spec) + epsilon))
#         spects.append((spec_db, audio[1]))
#     return spects

def gen_spectgrams(audio_data, max_signal_length, n_fft=128):
    spects = []
    count = 0
    for audio, sr, file in audio_data:
        padded_y = pad_segment(audio, max_signal_length)
        spec = stft(padded_y, frame_size=n_fft, overlap=n_fft//2)
        epsilon = 1e-4
        spec_db = np.array(20 * np.log10(np.abs(spec) + epsilon))
        spects.append((spec_db, sr, file))
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(audio_data)} spectrograms")
    return spects