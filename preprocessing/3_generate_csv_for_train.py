import os
import csv
import json
import random
from collections import Counter

min_file_num = 1000 #for class balance 

dataset_path= '/workspace/dataset/5.mat_for_train'
list_mat = []
imgkinds = ['LU','D']
imgtypes = ['RA','RE']

for imgkind in imgkinds:
    for imgtype in imgtypes:
        list_mat = []
        output_train_csv= f'/workspace/dataset/train_{imgkind}_{imgtype}.csv'        
        output_val_csv= f'/workspace/dataset/val_{imgkind}_{imgtype}.csv'
        key_counter = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                basefn = os.path.basename(file)
                if imgkind=='LU':
                    if basefn[0]=='L' and f'{imgtype}' in file and file.endswith(f'.mat'):
                        file_path = os.path.join(root, file)
                        list_mat.append(file_path)
                elif imgkind=='D':
                    if basefn[0]=='D' and f'{imgtype}' in file and file.endswith(f'.mat'):
                        file_path = os.path.join(root, file)
                        list_mat.append(file_path)

        key_counter = Counter("".join(file_path.split("_")[-2]) for file_path in list_mat)
        sorted_key_counter = dict(sorted(key_counter.items()))
        print(f'=============={imgkind}_{imgtype}==============')
        print(json.dumps(sorted_key_counter, indent=4, ensure_ascii=False))

        list_mat_selected = []
        for key, count in key_counter.items():            
            matching_paths = [file_path for file_path in list_mat if f'class_{int(key):02d}_' in file_path]

            if count <= min_file_num:
                list_mat_selected.extend(matching_paths)
            else:
                list_mat_selected.extend(random.sample(matching_paths, min_file_num))

        key_counter = Counter("".join(file_path.split("_")[-2]) for file_path in list_mat_selected)
        sorted_key_counter = dict(sorted(key_counter.items()))
        random.shuffle(list_mat_selected)

        ratio = 10
        index = 0

        with open(output_train_csv, mode='w', newline='') as train_file, open(output_val_csv, mode='w', newline='') as val_file:
            for line in list_mat_selected:        
                mat = line.replace("\n", "")                
                index+=1
                if index%ratio == 1:
                    writer = csv.writer(val_file)
                    writer.writerow([mat])
                else:
                    writer = csv.writer(train_file)
                    writer.writerow([mat])