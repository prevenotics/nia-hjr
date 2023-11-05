import os
import csv
import yaml
import random


with open('/root/work/hjr/IEEE_TGRS_SpectralFormer/config.yaml') as f:
        cfg = yaml.safe_load(f)

dataset_path= cfg['path']['dataset_path']
list_mat = []
imgtype = cfg['image_param']['type']

output_train_csv= f'train_{imgtype}.csv'
output_val_csv= f'val_{imgtype}.csv'
output_test_csv= f'test_{imgtype}.csv'

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.mat'):
            file_path = os.path.join(root, file)
            list_mat.append(file_path)

random.shuffle(list_mat)

ratio = 10
index = 0

with open(f'train_{imgtype}.csv', mode='w', newline='') as train_file, open(f'val_{imgtype}.csv', mode='w', newline='') as val_file, open(f'test_{imgtype}.csv', mode='w', newline='') as test_file:
    for line in list_mat:        
        mat = line.replace("\n", "")                
        index+=1
        if index%ratio == 0:
            writer = csv.writer(val_file)
            writer.writerow([mat])
        elif index%ratio == 1:
            writer = csv.writer(test_file)
            writer.writerow([mat])
        else:
            writer = csv.writer(train_file)
            writer.writerow([mat])