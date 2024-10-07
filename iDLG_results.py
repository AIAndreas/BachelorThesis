
import re
import numpy as np
import matplotlib.pyplot as plt


num_exp = 1000
dataset = "LFW"



"""
if dataset == "LFW":
    with open('Output_error/output_lfw_exp1000_utf8.txt', 'r', encoding='latin-1') as file:

if dataset == "LFW":
    with open('output_lfw_exp1000_utf8.txt', 'r', encoding='latin-1') as file:

        content = file.read()

else:
    # Open and read the file
    with open("Output_error/output_lfw_exp1000.txt", "r") as file:
        content = file.read()

"""

with open("Output_error/output_lfw_exp544_utf8_done.txt", "r") as file:
    content = file.read()


#cut first part of output 
content = content[content.find(f"running 0|{num_exp} experiment"):]



# Split the content at the first occurrence of '----------------------'
# Adjust this if there's a more specific marker
parts = content.split('----------------------')

print(parts[0])

#remove empty last element
parts = parts[:-1]


## LABEL ACCURACY 

#count number of correct answers
gt_label_count = 0
dlg_acc = 0
idlg_acc = 0 

for i in range(num_exp):

    gt_label_count += 1

    gt_label = parts[i].find("gt_label:")
    end = parts[i].find("lab_iDLG") #+12

    line = parts[i][gt_label:end+15]

    #line = parts[i][gt_label:end+12]
    print(line)
    # Split the line by whitespace or newlines
    line = re.split(r"[ \n\[\]]+", line)
    if line[3] == line[1]:
        dlg_acc += 1 
    
    if line[5] == line[1]:
        idlg_acc += 1
        

print("Number of correct labels for dlg: ", dlg_acc, ". Accuracy (%): ", (dlg_acc/num_exp)*100)
print("Number of correct labels for idlg: ", idlg_acc, ". Accuracy (%): ", (idlg_acc/num_exp)*100)

#gt_ label = 715 (+11)
#dlg_label = 727 (+23)
#idlg_label = 739 (+35


##MSE

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

#print(mse_iDLG)

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


import matplotlib.pyplot as plt

# Define thresholds (in decreasing order)
thresholds = [0.01, 0.005, 0.001, 0.0005, 0.0001]  # Corresponding thresholds

# Fidelity percentages (already computed)
fidelity_percentage_DLG = [f * 100 for f in fidelity_count_DLG]
fidelity_percentage_iDLG = [f * 100 for f in fidelity_count_iDLG]

# Plotting
plt.figure(figsize=(10, 8))

# DLG - blue line with circular markers
plt.plot(thresholds, fidelity_percentage_DLG, marker='o', color='blue', label='DLG', linestyle='-', markersize=10, linewidth = 3)

# iDLG - red line with star markers
plt.plot(thresholds, fidelity_percentage_iDLG, marker='*', color='red', label='iDLG', linestyle='-', markersize=12, linewidth = 3)

# Title and labels with increased font sizes
plt.title(dataset, fontsize=30)
plt.xlabel('Fidelity Threshold (MSE)', fontsize=40)
plt.ylabel('% of Good Fidelity', fontsize=40)

# Set x-axis to log scale to match the target plot and reverse the axis direction
plt.xscale('log')
plt.gca().invert_xaxis()  # Reverse the x-axis

# Customize the ticks on the x-axis to match the target plot (show exact thresholds instead of scientific notation)
plt.xticks([0.01, 0.005, 0.001, 0.0005, 0.0001], ['0.01', '0.005', '0.001', '0.0005', '0.0001'], fontsize=30)
plt.yticks([0,25,50,75,100],["0","25","50","75","100"], fontsize=30)

# Set y-axis limits from 0 to 100
plt.ylim(0, 100)

# Gridlines (optional, you can modify the linewidth if needed)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Legend with increased font size
plt.legend(fontsize=15)

# Show plot
#plt.show()



