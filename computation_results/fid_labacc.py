import chardet
import numpy as np
import matplotlib.pyplot as plt
import os
import chardet
import re
from utils_com import *

"""This code was designed by us, but have used ChatGPT for optimization, troubleshooting, and plots."""


## Convert to utf-8 files 

folder_name = "new_outputfiles/OG/"
under_folder = ["AMNIST/", "US/"]
output_path = "new_outputfiles/NEW/"

for i in under_folder:
    files = os.listdir(folder_name + i)
    for file_no in files: 
        with open(f'{folder_name}{i}{file_no}', 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            current_encoding = result['encoding']
        with open(f'{folder_name}{i}{file_no}', 'r', encoding=current_encoding) as file:
            content = file.read()

        if os.path.exists(output_path+i):
            for file in os.listdir(output_path+i):
                file_path = os.path.join(output_path+i, file_no)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

        with open(f'{output_path}{i}{file_no}', 'w', encoding='utf-8') as file:
            file.write(content)

        print(f"{file_no} successfully converted to UTF-8.")
# Define colors for each WS value
ws_color_mapping = {"512": "darkred", "1024": "darkgreen", "2048": "darkblue"}
thresholds = [0.01, 0.005, 0.001, 0.0005, 0.0001]
num_exp = 1000

# Plot function with label mismatch logging
def plot_fidelity_vs_threshold(path, dataset_name, ws_color_mapping):
    files = os.listdir(path)
    plt.figure(figsize=(10, 8))
    
    for filename in files:
        if not filename.endswith(".txt"):  # Skip non-txt files
            continue

        # Extract the WS number from the filename
        ws_number = filename.split('_')[-1].split('.')[0][4:]
        color = ws_color_mapping.get(ws_number, "black")  # Default to black if WS not found

        # Load content from file
        with open(f"{path}{filename}", "r") as file:
            content = file.read()

        # Ensure content has required sections before processing
        if f"running 0|{num_exp} experiment" not in content:
            print(f"Skipping {filename} as it does not contain expected data.")
            continue

        # Isolate relevant content section
        content = content[content.find(f"running 0|{num_exp} experiment"):]
        parts = content.split('----------------------')[:-1]

        # Check if parts contain data to avoid empty lists
        if not parts:
            print(f"No valid data sections found in {filename}, skipping this file.")
            continue

        # Track mismatches
        mismatched_labels = []

        # LABEL ACCURACY
        idlg_acc = 0
        for exp_num, part in enumerate(parts[:num_exp]):
            try:
                labels = re.split(r"[ \n\[\]]+", part[part.find("gt_label:"):part.find("lab_iDLG") + 15])
                if labels[3] == labels[1]:  # Comparing gt_label and lab_iDLG
                    idlg_acc += 1
                else:
                    mismatched_labels.append(exp_num)  # Track experiment numbers where labels differ
            except IndexError:
                print(f"Parsing error in file {filename}, experiment {exp_num}. Skipping this entry.")
                continue

        label_accuracy = (idlg_acc / num_exp) * 100 if num_exp > 0 else 0
        print(f"{dataset_name} - File {filename} - Label Accuracy (%): {label_accuracy:.2f}")

        # Print mismatches for this file if any
        if mismatched_labels:
            print(f"File {filename} - Label mismatches in experiments: {mismatched_labels}")

        # MSE Computation
        mse_iDLG = [
            float(mse_line[1]) if mse_line[1] != "inf" else np.inf
            for part in parts[:num_exp]
            for mse_line in [re.split(r"[ \n]+", part[part.find("mse_iDLG:"):part.find("gt_label")])]
        ]
        
        # Debugging step: Check what’s inside mse_iDLG
        print(f"File {filename} - Number of MSE values found: {len(mse_iDLG)}")
        if len(mse_iDLG) == 0:
            print(f"Warning: No MSE values found in file {filename}. Skipping this file.")
            continue
        
        # Fidelity counts at thresholds
        fidelity_counts = [sum(mse <= t for mse in mse_iDLG) / len(mse_iDLG) for t in thresholds]
        fidelity_percentage_iDLG = [f * 100 for f in fidelity_counts]

        # Plot each WS number's fidelity vs threshold
        plt.plot(thresholds, fidelity_percentage_iDLG, marker='*', linestyle='-', markersize=12, linewidth=3,
                 color=color, label=f"WS {ws_number}")

    # Final plot adjustments
    plt.title(dataset_name, fontsize=30)
    plt.xlabel('Fidelity Threshold (MSE)', fontsize=40)
    plt.ylabel('% of Good Fidelity', fontsize=40)
    plt.xscale('log')
    plt.gca().invert_xaxis()
    plt.xticks(thresholds, [str(t) for t in thresholds], fontsize=30)
    plt.yticks(range(0, 101, 25), map(str, range(0, 101, 25)), fontsize=30)
    plt.ylim(0, 100)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Modify legend position and size
    plt.legend(loc='lower left', fontsize=30)
    
    plt.show()

# Plot for AudioMNIST
plot_fidelity_vs_threshold("new_outputfiles/NEW/AMNIST/", "AudioMNIST", ws_color_mapping)

# Plot for UrbanSound8K
plot_fidelity_vs_threshold("new_outputfiles/NEW/US/", "UrbanSound8K", ws_color_mapping)
