import os
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy import io
import tifffile
import datetime
import yaml
import pandas as pd

image_folder = r'../../dataset/3.mat'
csv_path = ['train_mat_LU_RA.csv', 'train_mat_LU_RE.csv', 'train_mat_D_RA.csv', 'train_mat_D_RE.csv']
output_folder = image_folder.replace("3.mat","4.mat_for_train")

def sampling_point(image_size, y):
    image_width = image_size
    image_height = image_size

    center_x = image_width // 2
    center_y = image_height // 2

    if y == 1:
        sampling_coords = np.array([[center_x, center_y]])
    else:
        sub_divisions = y + 2
        x_coords = np.linspace(0, image_width, sub_divisions)
        y_coords = np.linspace(0, image_height, sub_divisions)

        x_mesh, y_mesh = np.meshgrid(x_coords[1:-1], y_coords[1:-1])
        sampling_coords = np.column_stack((x_mesh.ravel(), y_mesh.ravel())).astype(int)
    
    return sampling_coords

def main():
    
    
    os.makedirs(output_folder, exist_ok=True)
    # for i in range(len(csv_path)):
        # csv = pd.read_csv(csv_path[i], encoding='UTF-8')
    for csv in csv_path:
        if 'LU' in csv:
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
        
        
        each_csv = pd.read_csv(os.path.join(image_folder, csv), encoding='UTF-8')
        total_cnt = len(each_csv)
    
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        formatted_time_for_filename = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        # with open(f"log_{formatted_time_for_filename}.txt", "w") as log:
        #     log.write(f"Start Time [{formatted_time}]\n")
        print(f"Start Time [{formatted_time}]\n")
    
        num_classes = 31
        class_images = [[] for _ in range(num_classes)]
        size = 256
        images_per_file = size*size
        file_counter = [0] * num_classes
        
        for i in range(total_cnt):
            print(cnt)
            cnt +=1
            image_path = each_csv.iloc[i, 0]            
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

                    output_filename = os.path.join(output_folder, f'RE_class_{class_num:02d}_{file_counter[class_num]:07d}.mat')
                    io.savemat(output_filename, new_mat_file)
                    print(f'{cnt}/{total_cnt} : save: {output_filename}')
                    file_counter[class_num] += 1
        ########################################################################################################################
        # RA_class_counts = {f'{i:02d}': 0 for i in range(num_classes)}
        # RE_class_counts = {f'{i:02d}': 0 for i in range(num_classes)}

        # # RA_class_xx_xx.mat 파일용
        # for root, dirs, files in os.walk(output_folder):
        #     for file_name in files:
        #         if file_name.endswith('.mat'):
        #             file_name_only, _ = os.path.splitext(file_name)
        #             parts = file_name_only.split('_')
        #             class_prefix = parts[0]
        #             class_num = parts[2]                
        #             if class_prefix == 'RA':
        #                 RA_class_counts[class_num] += 1
        #             elif class_prefix == 'RE':
        #                 RE_class_counts[class_num] += 1
        

        # all_class_nums = list(RA_class_counts.keys() | RE_class_counts.keys())

        # for class_num in sorted(all_class_nums):
        #     count_RA = RA_class_counts.get(class_num, 0)
        #     count_RE = RE_class_counts.get(class_num, 0)
            
        #     special_mark = '*' if count_RA != count_RE else '' 
            
        #     print(f'Class {class_num}:  RA = {count_RA:06d}, RE = {count_RE:06d}  {special_mark}')
                           
        print(f"Total count =  {cnt}\n")
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")    
        print(f"End Time [{formatted_time}]\n")
                
        # with open(f"log_7_{formatted_time_for_filename}.txt", "a") as log:
        #     for class_num in sorted(all_class_nums):
        #         count_RA = RA_class_counts.get(class_num, 0)
        #         count_RE = RE_class_counts.get(class_num, 0)            
        #         special_mark = '*' if count_RA != count_RE else ' '             
        #         # log.write(f'Class {class_num}:  RA = {count_RA:06d}, RE = {count_RE:06d}  {special_mark}\n')
        #         log.write(f'Class {class_num}: {special_mark} RA = {count_RA}, RE = {count_RE}  \n')
            
        #     log.write(f"Total count = {total_cnt}\n")        
        #     log.write(f"Saved count = {cnt}\n")        
        #     log.write(f"End Time [{formatted_time}]\n")        
    
if __name__ == "__main__":
    main()