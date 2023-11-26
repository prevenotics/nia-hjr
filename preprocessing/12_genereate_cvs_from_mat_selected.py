import os
import csv
import yaml
import random
from collections import Counter

min_file_num = 120000000000

with open('/root/work/hjr/nia-hjr/cfg_test.yaml') as f:
    cfg = yaml.safe_load(f)

dataset_path= cfg['path']['dataset_path']
list_mat = []
imgtype = cfg['image_param']['type']

output_train_csv= f'list_{imgtype}.csv'
output_val_csv= f'val_drone_{imgtype}.csv'
output_test_csv= f'test_drone_{imgtype}.csv'


for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if f'{imgtype}' in file and file.endswith(f'.mat'):
            file_path = os.path.join(root, file)
            list_mat.append(file_path)

random.shuffle(list_mat)

# key_counter = Counter("".join(os.path.basename(file_path).split("_")[2]) for file_path in list_mat)
key_counter = Counter("".join(file_path.split("/")[-4]) for file_path in list_mat)
sorted_key_counter = dict(sorted(key_counter.items()))


list_mat_selected = []
for key, count in key_counter.items():
    matching_paths = [file_path for file_path in list_mat if f'/{key}/' in file_path]
    # matching_paths = [file_path for file_path in list_mat if f'class_{key}_' in file_path]
    # matching_paths = [file_path for file_path in list_mat if f'class_{int(key):02d}_' in file_path]

    if count <= min_file_num:
        list_mat_selected.extend(matching_paths)
    else:
        list_mat_selected.extend(random.sample(matching_paths, min_file_num))

key_counter = Counter("".join(file_path.split("/")[-4]) for file_path in list_mat_selected)
# key_counter = Counter("".join(os.path.basename(file_path).split("_")[2]) for file_path in list_mat_selected)
sorted_key_counter = dict(sorted(key_counter.items()))
random.shuffle(list_mat_selected)

ratio = 10
index = 0

with open(output_train_csv, mode='w', newline='') as train_file, open(output_val_csv, mode='w', newline='') as val_file, open(output_test_csv, mode='w', newline='') as test_file:
    for line in list_mat_selected:        
        mat = line.replace("\n", "")                
        index+=1
        # if index%ratio == 0:
        #     writer = csv.writer(val_file)
        #     writer.writerow([mat])
        # elif index%ratio == 1:
        #     writer = csv.writer(test_file)
        #     writer.writerow([mat])
        # else:
        writer = csv.writer(train_file)
        writer.writerow([mat])