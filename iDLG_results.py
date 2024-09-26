


# Open and read the file
with open("Output_error/Output_22780760.out", "r") as file:
    content = file.read()

num_exp = 1000

#cut first part of output 
content = content[content.find(f"running 0|{num_exp} experiment"):]

# Split the content at the first occurrence of '----------------------'
# Adjust this if there's a more specific marker
parts = content.split('----------------------')

#remove empty last element
parts = parts[:-1]

print(parts[0])

## LABEL ACCURACY 

#count number of correct answers
gt_label_count = 0
dlg_acc = 0
idlg_acc = 0 

for i in range(0,num_exp):
    gt_label = parts[i].find("gt_label:")
    gt_label_exp = parts[i][gt_label+11]
    gt_label_count += 1

    if parts[i][gt_label+23] == gt_label_exp:
        dlg_acc += 1
    
    if parts[i][gt_label+35] == gt_label_exp:
        idlg_acc += 1


print("Number of correct labels for dlg: ", dlg_acc, ". Accuracy (%): ", (dlg_acc/num_exp)*100)
print("Number of correct labels for idlg: ", idlg_acc, ". Accuracy (%): ", (idlg_acc/num_exp)*100)

#gt_ label = 715 (+11)
#dlg_label = 727 (+23)
#idlg_label = 739 (+35




##MSE

import re
import numpy as np
import matplotlib.pyplot as plt

# Initialize lists to store the MSE values
mse_DLG = []
mse_iDLG = []

count_dlg_inf = 0
count_idlg_inf = 0


for i in range(num_exp):
    start = parts[i].find("mse_DLG:")
    end = parts[i].find("gt_label")
    
    # Extract the relevant line from parts[i]
    line = parts[i][start:end]
    
    # Split the line by whitespace or newlines
    line = re.split(r"[ \n]+", line)

    if line[1] == "inf":
        mse_DLG.append(np.inf)
    else:
        mse_DLG.append(float(line[1]))

    if line[3] == "inf":
        mse_iDLG.append(np.inf)
    else:
        mse_iDLG.append(float(line[3]))

# Example print statements to verify the results (optional)
#print("mse_DLG:", mse_DLG)
#print("mse_iDLG:", mse_iDLG)

#print("inf dlg:", count_dlg_inf)
#print("inf idlg:", count_idlg_inf)


fidelity_count_DLG = []
fidelity_count_iDLG = []

if len(mse_DLG) == len(mse_iDLG):
    count_001_dlg = 0
    count_001_idlg = 0

    count_0005_dlg = 0
    count_0005_idlg = 0

    count_0001_dlg = 0
    count_0001_idlg = 0

    count_00005_dlg = 0
    count_00005_idlg = 0

    count_00001_dlg = 0
    count_00001_idlg = 0

    # Iterate through each mse value
    for i in range(len(mse_DLG)):
        if mse_DLG[i] <= 0.01:
            count_001_dlg += 1
        if mse_iDLG[i] <= 0.01:
            count_001_idlg += 1
        
        if mse_DLG[i] <= 0.005:
            count_0005_dlg += 1
        if mse_iDLG[i] <= 0.005:
            count_0005_idlg += 1
        
        if mse_DLG[i] <= 0.001:
            count_0001_dlg += 1
        if mse_iDLG[i] <= 0.001:
            count_0001_idlg += 1
        
        if mse_DLG[i] <= 0.0005:
            count_00005_dlg += 1
        if mse_iDLG[i] <= 0.0005:
            count_00005_idlg += 1
        
        if mse_DLG[i] <= 0.0001:
            count_00001_dlg += 1
        if mse_iDLG[i] <= 0.0001:
            count_00001_idlg += 1

    # Calculate fidelity for each threshold
    fidelity_count_DLG.append(count_001_dlg / len(mse_DLG))
    fidelity_count_iDLG.append(count_001_idlg / len(mse_iDLG))

    fidelity_count_DLG.append(count_0005_dlg / len(mse_DLG))
    fidelity_count_iDLG.append(count_0005_idlg / len(mse_iDLG))

    fidelity_count_DLG.append(count_0001_dlg / len(mse_DLG))
    fidelity_count_iDLG.append(count_0001_idlg / len(mse_iDLG))

    fidelity_count_DLG.append(count_00005_dlg / len(mse_DLG))
    fidelity_count_iDLG.append(count_00005_idlg / len(mse_iDLG))

    fidelity_count_DLG.append(count_00001_dlg / len(mse_DLG))
    fidelity_count_iDLG.append(count_00001_idlg / len(mse_iDLG))

# Now, fidelity_count_DLG and fidelity_count_iDLG contain the fidelity for each threshold

print(fidelity_count_DLG)
print(fidelity_count_iDLG)


# Define thresholds (in decreasing order)
thresholds = [0.01, 0.005, 0.001, 0.0005, 0.0001]  # Corresponding thresholds

# Convert fidelity counts to percentages
fidelity_percentage_DLG = [f * 100 for f in fidelity_count_DLG]
fidelity_percentage_iDLG = [f * 100 for f in fidelity_count_iDLG]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot with decreasing thresholds
plt.plot(thresholds, fidelity_percentage_DLG, marker='o', linestyle='-', color='blue', label='DLG Fidelity')
plt.plot(thresholds, fidelity_percentage_iDLG, marker='o', linestyle='-', color='orange', label='iDLG Fidelity')

# Invert the x-axis to represent decreasing threshold values
plt.gca().invert_xaxis()

# Set custom x-ticks for clearer spacing
plt.xticks(thresholds)

# Set custom y-ticks for percentages
percentage = [0, 25, 50, 75, 100]
plt.yticks(percentage)

# Disable minor ticks
plt.minorticks_off()

# Labels and title
plt.title('Fidelity Comparison for DLG and iDLG')
plt.xlabel('Threshold')
plt.ylabel('Fidelity (%)')
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.ylim(0, 100)  # Set y-axis limits from 0 to 100%

# Adjust x-axis limits to provide more space on the right side
plt.xlim(0.011, 0.00005)  # Extend the limits a bit on both sides

# Show the plot
#plt.show()