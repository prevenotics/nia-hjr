import os
import json
import numpy as np
# import rasterio
# from rasterio.transform import from_origin
from PIL import Image, ImageDraw
from scipy import io
import tifffile
import datetime
import pymysql
import yaml
import pandas as pd
# from pytz import timezone, utc
# KST = timezone('Asia/Seoul')
# now = datetime.datetime.utcnow()
# KST.localize(now)



image_folder = r'/root/work/hjr/dataset/3.mat'
output_folder = image_folder.replace("3.mat","4.mat")

# image_folder = r'/root/work/hjr/dataset/3.mat_test/'
# output_folder = image_folder.replace("3.mat_test/","4.mat_test/")

def sampling_point(image_size, y):
    # 이미지 크기
    image_width = image_size
    image_height = image_size

    # 중심 좌표 계산
    center_x = image_width // 2
    center_y = image_height // 2

    # y에 따른 샘플링 좌표 계산
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
    with open('/root/work/hjr/IEEE_TGRS_SpectralFormer/config.yaml') as f:
        cfg = yaml.safe_load(f)
    cnt = 0
    
    imgtype = cfg['image_param']['type']    
    band = cfg['image_param']['band']
    # csv_file = 
    #  = pd.read_csv(csv_file, encoding='UTF-8')        
        
    # sampling_coords = sampling_point(1024, 512)
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_time_for_filename = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    with open(f"log_7_{formatted_time_for_filename}.txt", "w") as log:
        log.write(f"Start Time [{formatted_time}]\n")
    
    for root, dirs, files in os.walk(image_folder):
        # if "02.수중 및 지상 초분광" in root:
            for image_filename in files:
                if image_filename.endswith('.mat'):
                    cnt +=1
    total_cnt = cnt
    cnt= 0
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")    
    print(f"Start Time [{formatted_time}]\n")
    
    
    # num_classes = cfg['num_class']
    num_classes = 31
    class_images = [[] for _ in range(num_classes)]
    size = 256
    images_per_file = size*size
    file_counter = [0] * num_classes
    
    for root, dirs, files in os.walk(image_folder):
        # if "02.수중 및 지상 초분광" in root:
            for image_filename in files:
                if image_filename.endswith(f'RE.mat'):
                    print(cnt)
                    cnt +=1
                    image_path = os.path.join(root, image_filename)                    
                    os.makedirs(output_folder, exist_ok=True)
                    
                    mat_file = io.loadmat(image_path)
                    image = mat_file['image']
                    label = mat_file['label']
                    
                    for class_num in range(num_classes):
                        class_indices = np.where(label == class_num)
                        class_images[class_num].extend(image[class_indices])
                        
                        while len(class_images[class_num]) >= images_per_file:
                             # 클래스별 이미지 데이터를 images_per_file 개수에 딱 맞춰서 저장
                            class_images_array = np.array(class_images[class_num][:images_per_file]).reshape(size, size, band)
                            class_images[class_num] = class_images[class_num][images_per_file:]

                            new_mat_file = {
                                'image': class_images_array,
                                'label': np.full((size, size), class_num)
                            }

                            output_filename = os.path.join(output_folder, f'RE_class_{class_num:02d}_{file_counter[class_num]:06d}.mat')
                            io.savemat(output_filename, new_mat_file)
                            print(f'{cnt}/{total_cnt} : save: {output_filename}')
                            file_counter[class_num] += 1

    for root, dirs, files in os.walk(image_folder):
        # if "02.수중 및 지상 초분광" in root:
            for image_filename in files:
                if image_filename.endswith(f'RA.mat'):
                    print(cnt)
                    cnt +=1
                    image_path = os.path.join(root, image_filename)                    
                    os.makedirs(output_folder, exist_ok=True)
                    
                    mat_file = io.loadmat(image_path)
                    image = mat_file['image']
                    label = mat_file['label']
                    
                    for class_num in range(num_classes):
                        class_indices = np.where(label == class_num)
                        class_images[class_num].extend(image[class_indices])
                        
                        while len(class_images[class_num]) >= images_per_file:
                             # 클래스별 이미지 데이터를 images_per_file 개수에 딱 맞춰서 저장
                            class_images_array = np.array(class_images[class_num][:images_per_file]).reshape(size, size, band)
                            class_images[class_num] = class_images[class_num][images_per_file:]

                            new_mat_file = {
                                'image': class_images_array,
                                'label': np.full((size, size), class_num)
                            }

                            output_filename = os.path.join(output_folder, f'RA_class_{class_num:02d}_{file_counter[class_num]:06d}.mat')
                            io.savemat(output_filename, new_mat_file)
                            print(f'{cnt}/{total_cnt} : save: {output_filename}')
                            file_counter[class_num] += 1                    
    #######################################################################################################################
    # size x size 미만으로 남은 것들 저장하기
    # for class_num in range(num_classes):
    #     if class_images[class_num]:
    #         remaining_images = class_images[class_num]
    #         num_remaining = len(remaining_images)
            
    #         if num_remaining < images_per_file:
    #             padding = np.zeros((images_per_file - num_remaining, band))
    #             remaining_images.extend(padding)
    #             label = np.full(num_remaining, class_num)
    #             label_ignore = np.full(images_per_file - num_remaining, 30)
    #             label = np.concatenate((label, label_ignore))
            
    #         class_images_array = np.array(remaining_images).reshape(size, size, band)
            
    #         new_mat_file = {
    #             'image': class_images_array,
    #             'label': label
    #         }

    #         output_filename = os.path.join(output_folder, f'{imgtype}_class_{class_num}_{file_counter[class_num]}.mat')
    #         io.savemat(output_filename, new_mat_file)
    #         print(f'save_{output_filename}')             
    #######################################################################################################################
    
    RA_class_counts = {f'{i:02d}': 0 for i in range(num_classes)}
    RE_class_counts = {f'{i:02d}': 0 for i in range(num_classes)}

    # RA_class_xx_xx.mat 파일용
    for root, dirs, files in os.walk(output_folder):
        for file_name in files:
            if file_name.endswith('.mat'):
                file_name_only, _ = os.path.splitext(file_name)
                parts = file_name_only.split('_')
                class_prefix = parts[0]
                class_num = parts[2]                
                if class_prefix == 'RA':
                    RA_class_counts[class_num] += 1
                elif class_prefix == 'RE':
                    RE_class_counts[class_num] += 1
    

    all_class_nums = list(RA_class_counts.keys() | RE_class_counts.keys())

    for class_num in sorted(all_class_nums):
        count_RA = RA_class_counts.get(class_num, 0)
        count_RE = RE_class_counts.get(class_num, 0)
        
        special_mark = '*' if count_RA != count_RE else '' 
        
        print(f'Class {class_num}:  RA = {count_RA:06d}, RE = {count_RE:06d}  {special_mark}')



    # RE.mat, RA.mat 파일용
    # for root, dirs, files in os.walk(image_folder):
    #     for file_name in files:
    #         if file_name.endswith('RE.mat'):
    #             file_name_only, _ = os.path.splitext(file_name)
    #             parts = root.split('/')
    #             # class_prefix = parts[0]
    #             class_num = parts[6]
    #             class_num = int(class_num.split('.')[0]) -1
    #             RE_class_counts[f'{class_num:02d}'] +=1
    #         if file_name.endswith('RA.mat'):
    #             file_name_only, _ = os.path.splitext(file_name)
    #             parts = root.split('/')                
    #             # class_prefix = parts[0]
    #             class_num = parts[6]
    #             class_num = int(class_num.split('.')[0]) -1
    #             RA_class_counts[f'{class_num:02d}'] +=1
                
    

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
            
    with open(f"log_7_{formatted_time_for_filename}.txt", "a") as log:
        for class_num in sorted(all_class_nums):
            count_RA = RA_class_counts.get(class_num, 0)
            count_RE = RE_class_counts.get(class_num, 0)            
            special_mark = '*' if count_RA != count_RE else ' '             
            # log.write(f'Class {class_num}:  RA = {count_RA:06d}, RE = {count_RE:06d}  {special_mark}\n')
            log.write(f'Class {class_num}: {special_mark} RA = {count_RA}, RE = {count_RE}  \n')
        
        log.write(f"Total count = {total_cnt}\n")        
        log.write(f"Saved count = {cnt}\n")        
        log.write(f"End Time [{formatted_time}]\n")        
    
if __name__ == "__main__":
    main()