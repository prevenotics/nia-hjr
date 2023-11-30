import os
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy import io
import tifffile
import datetime
import yaml
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='/workspace/dataset/4.mat')
args = parser.parse_args()


image_folder = args.path
csv_path = ['/workspace/dataset/4.mat/train_val_RA_for_class_mat.csv', '/workspace/dataset/4.mat/train_val_RE_for_class_mat.csv', '/workspace/dataset/4.mat/train_val_drone_RA_for_class_mat.csv', '/workspace/dataset/4.mat/train_val_drone_RE_for_class_mat.csv']
output_folder = image_folder.replace("4.mat","5.mat_for_train")


def main():
    
    
    os.makedirs(output_folder, exist_ok=True)
    
    for csv in csv_path:
        if 'drone' not in csv:
            if 'RA' in csv:
                imgkind = 'LU'
                imgtype = 'RA'
                band = 100
            else:
                imgkind = 'LU'
                imgtype = 'RE'
                band = 100
        else:
            if 'RA' in csv:
                imgkind = 'D'
                imgtype = 'RA'
                band = 80
            else:
                imgkind = 'D'
                imgtype = 'RE'
                band = 80
        
        with open(os.path.join(image_folder, csv), 'r', newline='') as csv_file:
            each_csv = [line.strip() for line in csv_file.readlines()]
        total_cnt = len(each_csv)
    
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_time_for_filename = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        
        print(f"Start Time [{formatted_time}]\n")
    
        num_classes = 31
        class_images = [[] for _ in range(num_classes)]
        size = 256
        images_per_file = size*size
        file_counter = [0] * num_classes
        cnt = 0
        for i in range(total_cnt):
            print(cnt)
            cnt +=1
            image_path = each_csv[i]            
            mat_file = io.loadmat(image_path)
            image = mat_file['image']
            label = mat_file['label']
            
            for class_num in range(num_classes):
                class_indices = np.where(label == class_num)
                class_images[class_num].extend(image[class_indices])
                
                while len(class_images[class_num]) >= images_per_file:                    
                    class_images_array = np.array(class_images[class_num][:images_per_file]).reshape(size, size, band)
                    class_images[class_num] = class_images[class_num][images_per_file:]

                    new_mat_file = {
                        'image': class_images_array,
                        'label': np.full((size, size), class_num)
                    }

                    output_filename = os.path.join(output_folder, f'{imgkind}_{imgtype}_class_{class_num:02d}_{file_counter[class_num]:010d}.mat')
                    io.savemat(output_filename, new_mat_file)
                    print(f'{cnt}/{total_cnt} : save: {output_filename}')
                    file_counter[class_num] += 1
        
                           
        print(f"Total count =  {cnt}\n")
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")    
        print(f"End Time [{formatted_time}]\n")
                
    
    
if __name__ == "__main__":
    main()