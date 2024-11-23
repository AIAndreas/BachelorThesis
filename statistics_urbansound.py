import numpy as np
import os
import librosa
import chardet
import re
from utils_com import *
import librosa.display
from scipy import stats



""" 
This code was designed by us, but have used ChatGPT for optimization, troubleshooting, and plots.

This code computes the accuracy measures for Urbansound8K. The statistical comparison is also included. 

Furthermore the comparison of each Window size to the baseline is included.

"""


def smse(xhat, x):
    # MSE Compute the (optimally scaled) mean squared error
    # between a (noisy) signal estimate and an original (noise free) signal
    #          _              _
    #         |             2  |
    # e = min | (x - a*xhat)   |
    #      a  |_              _|
    #
    # Usage
    #    e = mse(xhat, x);
    #
    # Inputs
    #    xhat    Estimated signal
    #    x       Original signal
    # 
    # Outputs
    #    e       Mean squared error
    #
    # Copyright 2013 Mikkel N. Schmidt, mnsc@dtu.dk
    # 2022, translated to Python by Tommy S. Alstrøm, tsal@dtu.dk

    a = (x.T @ xhat) / (xhat.T @ xhat)
    e = np.mean((x-a*xhat)**2)
    return e


def zip_and_sort(names, cum_avg):
    zipped = list(zip(names, cum_avg))
    zipped.sort(key=lambda x: x[0])
    sorted_names, sorted_cum_avg = zip(*zipped)
    return list(sorted_cum_avg)



#Load data

window_sizes = [512, 1024, 2048]


cum_avg_SMSE = {512: [], 1024: [], 2048: []}

name_recon_SMSE = {512: [], 1024: [], 2048: []}


#### ----------------------- BASELINE --------------------------------------

cum_avg_SMSE_baseline = {512: [], 1024: [], 2048: []}

name_recon_SMSE_baseline = {512: [], 1024: [], 2048: []}



for window_size in window_sizes:
    reconstructed_MNIST = f"results_audio/urbansound_results/urbansound{window_size}"
    MNIST_recon_files = os.listdir(reconstructed_MNIST)

    original_files_mnist = []

    for i in range(1,11):
        if i <= 10: 
            dir_og = f"original audio/Urbansound/fold{i}"
            dir_OG_files = os.listdir(dir_og)
            for file in dir_OG_files:
                if file in MNIST_recon_files:
                    original_files_mnist.append(file)


    #Sort away large MSE data 

    low_MSE_files_mnist = []


    # convert to utf-8
    MNIST_output_file_txt = f"new_outputfiles/NEW/US/output_urbansound_exp1000_nfft{window_size}.txt"
    with open(f'{MNIST_output_file_txt}', 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        current_encoding = result['encoding']
    with open(f'{MNIST_output_file_txt}', 'r', encoding=current_encoding) as file:
        content = file.read()
    with open(f'urbansound_output_exp1000_{window_size}.txt', 'w', encoding='utf-8') as file:
        file.write(content)


    # find file names and mse 
    num_exp = 1000
    with open(f"urbansound_output_exp1000_{window_size}.txt", "r") as file:
        content = file.read()

    content = content[content.find(f"running 0|{num_exp} experiment"):]
    parts = content.split('----------------------')
    parts = parts[:-1]


    for i in range(num_exp):
        mse_start = parts[i].find("mse_iDLG:")
        mse_end = parts[i].find("gt_label:")
        mse = parts[i][mse_start:mse_end]
        mse = re.split(r"[ \n\[\]]+", mse)

        if float(mse[1]) <= 0.01:
            start = parts[i].find("file name")
            end = parts[i].find("loss_iDLG:")
            line = parts[i][start:end]
            line = re.split(r"[ \n\[\]]+", line)
            file_name = re.split("\.", line[2])
            low_MSE_files_mnist.append(file_name[0]+".wav")

    # adjust lists
    MNIST_recon_files = [i for i in MNIST_recon_files if i in low_MSE_files_mnist]
    original_files_mnist = [i for i in original_files_mnist if i in low_MSE_files_mnist]



    # Base directory containing the folders
    audio_dir = "original audio/Urbansound/"
    recon_dir = f"results_audio/urbansound_results/urbansound{window_size}/"

    # List of all folders ("fold1", "fold2", ..., "fold10")
    folders = [f"fold{i}" for i in range(1, 11)]

    # List of files to find and process
    all_files = MNIST_recon_files  # Replace with your list of file names


    # SMSE
    for file_recon in MNIST_recon_files:
        name_recon_SMSE[window_size].append(file_recon)


        file_found = False  # Flag to indicate if the file is found in the original folders

        # Check each folder for the original file
        for folder in folders:
            folder_path = os.path.join(audio_dir, folder)
            file_path = os.path.join(folder_path, file_recon)

            # If the file exists, process it
            if os.path.exists(file_path):


                # Load the reconstructed and original (clean) audio files
                y_hat, sr_hat = librosa.load(os.path.join(recon_dir, file_recon), sr=None)
                clean, sr_y = librosa.load(file_path, sr=None)

                # Calculate the difference in length between the original and reconstructed signals
                len_diff = len(y_hat) - len(clean)

                # If y_hat is longer, trim it
                if len_diff > 0:
                    # Calculate the number of samples to remove from each end
                    trim_amount = len_diff // 2
                    y_hat = y_hat[trim_amount:-trim_amount]  # Trim both ends
                elif len_diff < 0:
                    trim_amount = -len_diff // 2
                    clean = clean[trim_amount:-trim_amount]  # Trim both ends of the original signal

                # Ensure both y_hat and clean are the same length after trimming
                min_len = min(len(y_hat), len(clean))
                y_hat = y_hat[:min_len]
                clean = clean[:min_len]

                # Normalize both signals
                normalized_clean = clean / np.sqrt(np.sum(clean**2))
                normalized_yhat = y_hat / np.sqrt(np.sum(y_hat**2))

                # Calculate SMSE
                e = smse(y_hat, clean)
                cum_avg_SMSE[window_size].append(e)

                file_found = True
                break  # Exit the loop once the file is found and processed

        # If the file was not found in any folder, handle the missing file
        if not file_found:
            print(f"Original file for {file_recon} not found in any folder.")








#### ----------------------- BASELINE --------------------------------------

    og_dir = "original audio/Urbansound/"
    recon_dir = f"baseline_recon/urbansound{window_size}/"


    # SMSE
    for file_recon in MNIST_recon_files:
        name_recon_SMSE_baseline[window_size].append(file_recon)

        file_found = False  # Flag to indicate if the file is found in the original folders

        # Check each folder for the original file
        for folder in folders:
            folder_path = os.path.join(og_dir, folder)
            file_path = os.path.join(folder_path, file_recon)

            # If the file exists, process it
            if os.path.exists(file_path):

                # Load the reconstructed and original (clean) audio files
                y_hat, sr_hat = librosa.load(os.path.join(recon_dir, file_recon), sr=None)
                y, sr_y = librosa.load(file_path, sr=None)

                # Calculate the difference in length between the original and reconstructed signals
                len_diff = len(y_hat) - len(y)

                # If y_hat is longer, trim it
                if len_diff > 0:
                    # Calculate the number of samples to remove from each end
                    trim_amount = len_diff // 2
                    y_hat = y_hat[trim_amount:-trim_amount]  # Trim both ends
                elif len_diff < 0:
                    trim_amount = -len_diff // 2
                    y = y[trim_amount:-trim_amount]  # Trim both ends of the original signal

                # Ensure both y_hat and y are the same length after trimming
                min_len = min(len(y_hat), len(y))
                y_hat = y_hat[:min_len]
                y = y[:min_len]

                # Normalize the signals
                normalized_y = y / np.sqrt(np.sum(y**2))
                normalized_yhat = y_hat / np.sqrt(np.sum(y_hat**2))

                # Compute SMSE
                e = smse(y_hat, y)
                cum_avg_SMSE_baseline[window_size].append(e)

                file_found = True
                break  # Exit the loop once the file is found and processed

        # If the file was not found in any folder, handle the missing file
        if not file_found:
            print(f"Original file for {file_recon} not found in any folder.")



for window_size in window_sizes:
    print(f"Average SMSE for {window_size}:", np.average(cum_avg_SMSE[window_size]))

print("---")

""" Statistical tests commence here:"""


# Test for normality of each accuracy measure and window size 


for window_size in window_sizes:
    result = stats.shapiro(cum_avg_SMSE[window_size])
    if result.pvalue >= 0.05:
        print(f"{window_size} window p-value (SMSE): {result.pvalue}") 

print("---")



### Because none of the tests have three groups that are normally distributed, we use Kruskil-Wallis 

# SMSE
smse_krus = stats.kruskal(cum_avg_SMSE[512], cum_avg_SMSE[1024], cum_avg_SMSE[2048])
print("The p-value for KW test for SMSE:", smse_krus.pvalue)
print("---")




#### ----------------------- BASELINE --------------------------------------

print("\n ----------------------- BASELINE -------------------------------------- \n")


for window_size in window_sizes:
    print(f"Average SMSE for {window_size} BL:", np.average(cum_avg_SMSE_baseline[window_size]))

print("---")


for window_size in window_sizes:
    result = stats.shapiro(cum_avg_SMSE_baseline[window_size])
    if result.pvalue >= 0.05:
        print(f"{window_size} window p-value (SMSE) BL: {result.pvalue}") 

smse_krus_BL = stats.kruskal(cum_avg_SMSE_baseline[512], cum_avg_SMSE_baseline[1024], cum_avg_SMSE_baseline[2048])
print("The p-value for KW test for SMSE BL:", smse_krus_BL.pvalue)
print("---")


# WILCOXON SMSE
recon_cum_avg_sorted_SMSE = {}
baseline_cum_avg_sorted_SMSE = {}

for window_size in [512, 1024, 2048]:
    recon_cum_avg_sorted_SMSE[window_size] = zip_and_sort(name_recon_SMSE[window_size], cum_avg_SMSE[window_size])
    baseline_cum_avg_sorted_SMSE[window_size] = zip_and_sort(name_recon_SMSE_baseline[window_size], cum_avg_SMSE_baseline[window_size])


for window_size in [512, 1024, 2048]:
    recon_data = recon_cum_avg_sorted_SMSE[window_size]
    baseline_data = baseline_cum_avg_sorted_SMSE[window_size]

    stat, p_value = stats.wilcoxon(recon_data, baseline_data)
    N = len(recon_data)

    z_score = (stat - (N * (N + 1)) / 4) / np.sqrt(N * (N + 1) * (2 * N + 1) / 24)
    
    effect_size = z_score / np.sqrt(N)
    
    # Print the result
    print(f"Window size {window_size} SMSE:")
    print(f"  Wilcoxon statistic: {stat}")
    print(f"  p-value: {p_value}")
    print(f"  z-score: {z_score}")
    print(f"  effect size: {effect_size}\n")

