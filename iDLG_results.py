


# Open and read the file
with open("Output_error/Output_22772684.out", "r") as file:
    content = file.read()

num_exp = 100

#cut first part of output 
content = content[content.find(f"running 0|{num_exp} experiment"):]

# Split the content at the first occurrence of '----------------------'
# Adjust this if there's a more specific marker
parts = content.split('----------------------')

#remove empty last element
parts = parts[:-1]



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
#idlg_label = 739 (+35)


print(parts[0])