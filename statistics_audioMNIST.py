import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import soundfile as sf 
import chardet
import re
from utils_com import *
import librosa.display
from pystoi import stoi
from scipy.stats import levene
from scipy import stats


""" 
This code was designed by us, but have used ChatGPT for optimization, troubleshooting, and plots. 

This code computes the accuracy measures for AudioMNIST. The statistical comparison is also included. 

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

cum_avg_RMSE = {512: [], 1024: [], 2048: []}
cum_avg_SMSE = {512: [], 1024: [], 2048: []}
cum_avg_STOI = {512: [], 1024: [], 2048: []}


name_recon_RMSE = {512: [], 1024: [], 2048: []}
name_recon_SMSE = {512: [], 1024: [], 2048: []}
name_recon_STOI = {512: [], 1024: [], 2048: []}

#### ----------------------- BASELINE --------------------------------------
cum_avg_RMSE_baseline = {512: [], 1024: [], 2048: []}
cum_avg_SMSE_baseline = {512: [], 1024: [], 2048: []}
cum_avg_STOI_baseline = {512: [], 1024: [], 2048: []}

name_recon_RMSE_baseline = {512: [], 1024: [], 2048: []}
name_recon_SMSE_baseline = {512: [], 1024: [], 2048: []}
name_recon_STOI_baseline = {512: [], 1024: [], 2048: []}


for window_size in window_sizes:
    reconstructed_MNIST = f"results_audio/audiomnist_results/audio_mnist{window_size}"
    MNIST_recon_files = os.listdir(reconstructed_MNIST)

    original_files_mnist = []

    for i in range(1,61):
        if i < 10: 
            dir_og = f"original audio/AMNISTdata/0{i}"
            dir_OG_files = os.listdir(dir_og)
            for file in dir_OG_files:
                if file in MNIST_recon_files:
                    original_files_mnist.append(file)
        else:
            dir_og = f"original audio/AMNISTdata/{i}"
            dir_OG_files = os.listdir(dir_og)
            for file in dir_OG_files:
                if file in MNIST_recon_files:
                    original_files_mnist.append(file)


    #Sort away large MSE data 

    low_MSE_files_mnist = []


    # convert to utf-8
    MNIST_output_file_txt = f"new_outputfiles/NEW/AMNIST/output_audiomnist_exp1000_nfft{window_size}.txt"
    with open(f'{MNIST_output_file_txt}', 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        current_encoding = result['encoding']
    with open(f'{MNIST_output_file_txt}', 'r', encoding=current_encoding) as file:
        content = file.read()
    with open(f'audiomnist_output_exp1000_{window_size}.txt', 'w', encoding='utf-8') as file:
        file.write(content)


    # find file names and mse 
    num_exp = 1000
    with open(f"audiomnist_output_exp1000_{window_size}.txt", "r") as file:
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


    og_dir = "original audio/AMNISTdata/"
    recon_dir = f"results_audio/audiomnist_results/audio_mnist{window_size}/"

    #STOI
    for file_recon in MNIST_recon_files:
        name_recon_STOI[window_size].append(file_recon)

        y_hat = file_recon
        y = [i for i in original_files_mnist if i in y_hat]
        denoised, fs = sf.read(recon_dir + y_hat)

        file_no = re.split("\_", y[0])[1]
        clean, fs = sf.read(og_dir + file_no + "/" + y[0])


        #trim signal
        len_diff = len(denoised) - len(clean)
        if len_diff > 0:
            trim_amount = len_diff // 2
            denoised = denoised[trim_amount:-trim_amount] 
        elif len_diff < 0:
            trim_amount = -len_diff // 2
            clean = clean[trim_amount:-trim_amount]  

        #ensure both y_hat and y are the same length after trimming
        min_len = min(len(denoised), len(clean))
        denoised = denoised[:min_len]
        clean = clean[:min_len]


        # Clean and den should have the same length, and be 1D
        d = stoi(clean, denoised, fs, extended=False)
        cum_avg_STOI[window_size].append(d)


    #RMSE 
    counter = 0

    for file_recon in MNIST_recon_files:
        name_recon_RMSE[window_size].append(file_recon)
        counter += 1
        y_hat = file_recon
        y = [i for i in original_files_mnist if i in y_hat]
        y_hat, sr_hat = librosa.load(recon_dir + y_hat, sr = None)
        file_no = re.split("\_", y[0])[1]
        y, sr_y = librosa.load(og_dir + file_no + "/" + y[0], sr = None)
        # Calculate the difference in length between the original and reconstructed signals
        len_diff = len(y_hat) - len(y)
        #print(len_diff)
        # If y_recon is longer, trim it
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
        normalized_y = y / np.sqrt(np.sum(y**2))
        normalized_yhat = y_hat / np.sqrt(np.sum(y_hat**2))

        RMSE = np.sqrt((np.sum((normalized_yhat-normalized_y)**2))/len(normalized_y))
        cum_avg_RMSE[window_size].append(RMSE)


    #SMSE
    for file_recon in MNIST_recon_files:
        name_recon_SMSE[window_size].append(file_recon)
        counter += 1
        y_hat = file_recon
        y = [i for i in original_files_mnist if i in y_hat]
        y_hat, sr_hat = librosa.load(recon_dir + y_hat, sr = None)
        file_no = re.split("\_", y[0])[1]
        y, sr_y = librosa.load(og_dir + file_no + "/" + y[0], sr = None)
        # Calculate the difference in length between the original and reconstructed signals
        len_diff = len(y_hat) - len(y)
        #print(len_diff)
        # If y_recon is longer, trim it
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
        normalized_y = y / np.sqrt(np.sum(y**2))
        normalized_yhat = y_hat / np.sqrt(np.sum(y_hat**2))

        e = smse(y_hat, y)
        
        cum_avg_SMSE[window_size].append(e)








#### ----------------------- BASELINE --------------------------------------

    og_dir = "original audio/AMNISTdata/"
    recon_dir = f"baseline_recon/audio_mnist{window_size}/"

    #STOI
    for file_recon in MNIST_recon_files:
        name_recon_STOI_baseline[window_size].append(file_recon)

        y_hat = file_recon
        y = [i for i in original_files_mnist if i in y_hat]
        denoised, fs = sf.read(recon_dir + y_hat)

        file_no = re.split("\_", y[0])[1]
        clean, fs = sf.read(og_dir + file_no + "/" + y[0])


        #trim signal
        len_diff = len(denoised) - len(clean)
        if len_diff > 0:
            trim_amount = len_diff // 2
            denoised = denoised[trim_amount:-trim_amount] 
        elif len_diff < 0:
            trim_amount = -len_diff // 2
            clean = clean[trim_amount:-trim_amount]  

        #ensure both y_hat and y are the same length after trimming
        min_len = min(len(denoised), len(clean))
        denoised = denoised[:min_len]
        clean = clean[:min_len]


        # Clean and den should have the same length, and be 1D
        d = stoi(clean, denoised, fs, extended=False)
        cum_avg_STOI_baseline[window_size].append(d)


    #RMSE 
    counter = 0

    for file_recon in MNIST_recon_files:
        name_recon_RMSE_baseline[window_size].append(file_recon)
        counter += 1
        y_hat = file_recon
        y = [i for i in original_files_mnist if i in y_hat]
        y_hat, sr_hat = librosa.load(recon_dir + y_hat, sr = None)
        file_no = re.split("\_", y[0])[1]
        y, sr_y = librosa.load(og_dir + file_no + "/" + y[0], sr = None)
        # Calculate the difference in length between the original and reconstructed signals
        len_diff = len(y_hat) - len(y)
        #print(len_diff)
        # If y_recon is longer, trim it
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
        normalized_y = y / np.sqrt(np.sum(y**2))
        normalized_yhat = y_hat / np.sqrt(np.sum(y_hat**2))

        RMSE = np.sqrt((np.sum((normalized_yhat-normalized_y)**2))/len(normalized_y))
        cum_avg_RMSE_baseline[window_size].append(RMSE)


    #SMSE
    for file_recon in MNIST_recon_files:
        name_recon_SMSE_baseline[window_size].append(file_recon)
        counter += 1
        y_hat = file_recon
        y = [i for i in original_files_mnist if i in y_hat]
        y_hat, sr_hat = librosa.load(recon_dir + y_hat, sr = None)
        file_no = re.split("\_", y[0])[1]
        y, sr_y = librosa.load(og_dir + file_no + "/" + y[0], sr = None)
        # Calculate the difference in length between the original and reconstructed signals
        len_diff = len(y_hat) - len(y)
        #print(len_diff)
        # If y_recon is longer, trim it
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
        normalized_y = y / np.sqrt(np.sum(y**2))
        normalized_yhat = y_hat / np.sqrt(np.sum(y_hat**2))

        e = smse(y_hat, y)
        
        cum_avg_SMSE_baseline[window_size].append(e)


for window_size in window_sizes:
    print(f"Average RMSE for {window_size}:", np.average(cum_avg_RMSE[window_size]))

print("---")

for window_size in window_sizes:
    print(f"Average SMSE for {window_size}:", np.average(cum_avg_SMSE[window_size]))

print("---")

for window_size in window_sizes:
    print(f"Average STOI for {window_size}:", np.average(cum_avg_STOI[window_size]))

print("---")

""" Statistical tests commence here:"""


# Test for normality of each accuracy measure and window size 


for window_size in window_sizes:
    result = stats.shapiro(cum_avg_RMSE[window_size])
    if result.pvalue >= 0.05:
        print(f"{window_size} window p-value (RMSE): {result.pvalue}") 

print("---")

for window_size in window_sizes:
    result = stats.shapiro(cum_avg_SMSE[window_size])
    if result.pvalue >= 0.05:
        print(f"{window_size} window p-value (SMSE): {result.pvalue}") 

print("---")

for window_size in window_sizes:
    result = stats.shapiro(cum_avg_STOI[window_size])
    if result.pvalue >= 0.05:
        print(f"{window_size} window p-value (STOI): {result.pvalue}") 

print("---")


### Because none of the tests have three groups that are normally distributed, we use Kruskil-Wallis 


# RMSE 
rmse_krus = stats.kruskal(cum_avg_RMSE[512], cum_avg_RMSE[1024], cum_avg_RMSE[2048])
print("The p-value for KW test for RMSE:", rmse_krus.pvalue)
print("---")
# SMSE
smse_krus = stats.kruskal(cum_avg_SMSE[512], cum_avg_SMSE[1024], cum_avg_SMSE[2048])
print("The p-value for KW test for SMSE:", smse_krus.pvalue)
print("---")
# STOI
stoi_krus = stats.kruskal(cum_avg_STOI[512], cum_avg_STOI[1024], cum_avg_STOI[2048])
print("The p-value for KW test for STOI:", stoi_krus.pvalue)
print("---")


# Further KW tests for STOI 
stoi_krus_1 = stats.kruskal(cum_avg_STOI[512], cum_avg_STOI[1024])
print("The p-value for KW test for STOI (512-1024):", stoi_krus_1.pvalue)

stoi_krus_2 = stats.kruskal(cum_avg_STOI[512], cum_avg_STOI[2048])
print("The p-value for KW test for STOI (512-2048):", stoi_krus_2.pvalue)

stoi_krus_3 = stats.kruskal(cum_avg_STOI[1024], cum_avg_STOI[2048])
print("The p-value for KW test for STOI (1024-2048):", stoi_krus_3.pvalue)


# Multiple box plots on one Axes
fig, ax = plt.subplots()
ax.boxplot([cum_avg_STOI[512], cum_avg_STOI[1024], cum_avg_STOI[2048]])
ax.set_xticklabels(['512', '1024', '2048'])
#plt.show()



#### ----------------------- BASELINE --------------------------------------

print("\n ----------------------- BASELINE -------------------------------------- \n")

for window_size in window_sizes:
    print(f"Average RMSE for {window_size} BL:", np.average(cum_avg_RMSE_baseline[window_size]))

print("---")

for window_size in window_sizes:
    print(f"Average SMSE for {window_size} BL:", np.average(cum_avg_SMSE_baseline[window_size]))

print("---")

for window_size in window_sizes:
    print(f"Average STOI for {window_size} BL:", np.average(cum_avg_STOI_baseline[window_size]))

print("---")


for window_size in window_sizes:
    result = stats.shapiro(cum_avg_RMSE_baseline[window_size])
    if result.pvalue >= 0.05:
        print(f"{window_size} window p-value (RMSE) BL: {result.pvalue}") 

print("---")

for window_size in window_sizes:
    result = stats.shapiro(cum_avg_SMSE_baseline[window_size])
    if result.pvalue >= 0.05:
        print(f"{window_size} window p-value (SMSE) BL: {result.pvalue}") 

print("---")

for window_size in window_sizes:
    result = stats.shapiro(cum_avg_STOI_baseline[window_size])
    if result.pvalue >= 0.05:
        print(f"{window_size} window p-value (STOI) BL: {result.pvalue}") 

print("---")






# WILCOXON RMSE
recon_cum_avg_sorted_RMSE = {}
baseline_cum_avg_sorted_RMSE = {}

for window_size in [512, 1024, 2048]:
    recon_cum_avg_sorted_RMSE[window_size] = zip_and_sort(name_recon_RMSE[window_size], cum_avg_RMSE[window_size])
    baseline_cum_avg_sorted_RMSE[window_size] = zip_and_sort(name_recon_RMSE_baseline[window_size], cum_avg_RMSE_baseline[window_size])


for window_size in [512, 1024, 2048]:
    recon_data = recon_cum_avg_sorted_RMSE[window_size]
    baseline_data = baseline_cum_avg_sorted_RMSE[window_size]

    stat, p_value = stats.wilcoxon(recon_data, baseline_data)
    N = len(recon_data)

    z_score = (stat - (N * (N + 1)) / 4) / np.sqrt(N * (N + 1) * (2 * N + 1) / 24)
    
    effect_size = z_score / np.sqrt(N)
    
    # Print the result
    print(f"Window size {window_size} RMSE:")
    print(f"  Wilcoxon statistic: {stat}")
    print(f"  p-value: {p_value}")
    print(f"  z-score: {z_score}")
    print(f"  effect size: {effect_size}\n")




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


# WILCOXON STOI
recon_cum_avg_sorted_STOI = {}
baseline_cum_avg_sorted_STOI = {}

for window_size in [512, 1024, 2048]:
    recon_cum_avg_sorted_STOI[window_size] = zip_and_sort(name_recon_STOI[window_size], cum_avg_STOI[window_size])
    baseline_cum_avg_sorted_STOI[window_size] = zip_and_sort(name_recon_STOI_baseline[window_size], cum_avg_STOI_baseline[window_size])


for window_size in [512, 1024, 2048]:
    recon_data = recon_cum_avg_sorted_STOI[window_size]
    baseline_data = baseline_cum_avg_sorted_STOI[window_size]

    stat, p_value = stats.wilcoxon(recon_data, baseline_data)
    N = len(recon_data)

    z_score = (stat - (N * (N + 1)) / 4) / np.sqrt(N * (N + 1) * (2 * N + 1) / 24)
    
    effect_size = z_score / np.sqrt(N)
    
    # Print the result
    print(f"Window size {window_size} STOI:")
    print(f"  Wilcoxon statistic: {stat}")
    print(f"  p-value: {p_value}")
    print(f"  z-score: {z_score}")
    print(f"  effect size: {effect_size}\n")