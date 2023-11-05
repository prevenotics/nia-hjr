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


image_folder = r'/root/work/hjr/dataset/3.mat/'
# json_folder = r'D:/DATA/hjr/dataset/2.라벨링데이터/'
# output_image_folder = r'D:/DATA/hjr/dataset/1.image_mat/'
# output_json_folder = r'D:/DATA/hjr/dataset/2.label_mat/'
# output_mat_folder = r'D:/DATA/hjr/dataset/3.mat/'

# input_path= input("input path : ")
# image_folder = input_path

output_mat_folder = image_folder.replace("3.mat","3.final_mat")

def sampling_point(image_size, y):
    # 이미지 크기
    image_width = image_size
    image_height = image_size

    # y 변수 설정
    y = 512  # y 값을 변경하여 원하는 등분 수를 얻을 수 있습니다.

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
    cnt = 0
    file_paths_dict = {}
    imgtype = ["RA", "RE"]
    band = 100
    # band = 120
    
    sampling_coords = sampling_point(1024, 512)
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "w") as log:
        log.write(f"Start Time [{formatted_time}]\n")
    
    for root, dirs, files in os.walk(image_folder):
        # if "02.수중 및 지상 초분광" in root:
            for image_filename in files:
                if image_filename.endswith('.mat'):
                    cnt +=1
    
    total_cnt = cnt
    cnt= 0
    error_cnt=0
    
    for root, dirs, files in os.walk(image_folder):
        # if "02.수중 및 지상 초분광" in root:
            for image_filename in files:
                if image_filename.endswith('.mat'):
                    base_name, file_extension = os.path.splitext(image_filename)
                    ####################################################################################                    
                    # prefix = '_'.join(image_filename.split('_')[:-1])  # Extract the common prefix

                    # if prefix not in file_paths_dict:
                    #     file_paths_dict[prefix] = []
                    # else:
                    #     continue
                    ####################################################################################
                                        
                    image_path = os.path.join(root, f"{prefix}{file_extension}")
                    label_path = image_path.replace("1.원천데이터","2.라벨링데이터")
                    relative_path = os.path.relpath(root, image_folder)
                    out_mat_folder = os.path.join(output_mat_folder, relative_path)                   
                    os.makedirs(out_mat_folder, exist_ok=True)                    
                    mat_path_RA = os.path.join(out_mat_folder, f"{prefix}_RA.mat")
                    mat_path_RE = os.path.join(out_mat_folder, f"{prefix}_RE.mat")
                    
                    
                    # tif_path_RA = os.path.join(out_mat_folder, f"{prefix}_RA.tif")
                    # tif_path_RE = os.path.join(out_mat_folder, f"{prefix}_RE.tif")
                    try:
                        if prefix[2] == 'L':
                            print(f"{cnt}/{int(total_cnt/40)}({total_cnt})\t{image_path}\n") 
                            mat_image_RA = L_file(image_path, imgtype[0])
                            mat_image_RE = L_file(image_path, imgtype[1])
                            label_path = label_path.replace(".tif", "_RE01.json")
                            mat_label=create_label_mat(label_path, prefix[2], sampling_coords)
                        elif prefix[2] == 'U':                        
                            print(f"{cnt}/{int(total_cnt/30)}({total_cnt})\t{image_path}\n") 
                            mat_image_RA = U_file(image_path, imgtype[0], sampling_coords)
                            mat_image_RE = U_file(image_path, imgtype[1], sampling_coords)                        
                            label_path = label_path.replace(".tif", "_RE21.json")
                            mat_label=create_label_mat(label_path, prefix[2], sampling_coords)
                        elif prefix[2] == 'D':
                            mat_image_RA = D_file(image_path, imgtype[0])
                            mat_image_RE = D_file(image_path, imgtype[1])                        
                            label_path = label_path.replace(".tif", "_RE36.json")
                            mat_label=create_label_mat(label_path, prefix[2], sampling_coords)
                        
                        io.savemat(mat_path_RA, {'image': np.array(mat_image_RA), 'label' : np.array(mat_label)})
                        io.savemat(mat_path_RE, {'image': np.array(mat_image_RE), 'label' : np.array(mat_label)})
                    except FileNotFoundError as e:
                        error_cnt +=1
                        current_time = datetime.datetime.now()
                        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        with open("log.txt", "a") as log:
                            log.write(f"{formatted_time} : [FileNotFoundError]\t{image_path}\n")
                            
                        continue
                                        
                    # io.savemat(mat_path_RA, {'image': np.array(mat_image_RA), 'label' : np.array(mat_label)})
                    # io.savemat(mat_path_RE, {'image': np.array(mat_image_RE), 'label' : np.array(mat_label)})
                    # tifffile.imsave(tif_path_RA, np.array(mat_image_RA))
                    # tifffile.imsave(tif_path_RE, np.array(mat_image_RE))

                    cnt +=1
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as log:
        log.write(f"Total count = {cnt}\n")
        log.write(f"Error count = {error_cnt}\n")        
        log.write(f"End Time [{formatted_time}]\n")        
    print(f"Total count =  {cnt}\n")
    print(f"Error count = {error_cnt}\n")


        
    
if __name__ == "__main__":
    main()