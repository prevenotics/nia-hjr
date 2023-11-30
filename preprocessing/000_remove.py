import os
os.chdir("/workspace/nia-hjr/preprocessing")

# Read the contents of LU_RE.txt and remove.txt
with open('LU_RE_100p.txt', 'r') as lu_re_file:
    lu_re_lines = lu_re_file.readlines()

with open('remove.txt', 'r') as remove_file:
    remove_lines = remove_file.readlines()

# Extract filenames from remove.txt
files_to_remove = [line.strip() for line in remove_lines]

# Filter out lines containing filenames from remove.txt
filtered_lines = [line for line in lu_re_lines if not any(file_name in line for file_name in files_to_remove)]

# Write the filtered content back to LU_RE.txt
with open('LU_RE.txt', 'w') as lu_re_file:
    lu_re_file.writelines(filtered_lines)
