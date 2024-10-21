import chardet

# Step 1: Detect the encoding of the original file
with open('Output_error/output_acoustsig_non_opt_exp1000.txt', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    current_encoding = result['encoding']
    print(f"Detected encoding: {current_encoding}")

# Step 2: Read the file with the detected encoding
with open('Output_error/output_acoustsig_non_opt_exp1000.txt', 'r', encoding=current_encoding) as file:
    content = file.read()

# Step 3: Write the content to a new file in UTF-8 encoding
with open('output_acoustsig_non_opt_exp1000_utf8.txt', 'w', encoding='utf-8') as file:
    file.write(content)

print("File successfully converted to UTF-8.")
