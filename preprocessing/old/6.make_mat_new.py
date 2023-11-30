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


image_folder = r'/workspace/dataset/1.원천데이터_temp/'
# json_folder = r'D:/DATA/hjr/dataset/2.라벨링데이터/'
# output_image_folder = r'D:/DATA/hjr/dataset/1.image_mat/'
# output_json_folder = r'D:/DATA/hjr/dataset/2.label_mat/'
# output_mat_folder = r'D:/DATA/hjr/dataset/3.mat/'

# input_path= input("input path : ")
# image_folder = input_path

output_mat_folder = image_folder.replace("1.원천데이터","3.mat_temp")

def create_label_mat(label_path, loc, sampling_coords, output_size):
    try:    
        with open(label_path, "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError as e:
        raise e
    
    image_id = data["image"][0]["id"]
    image_width = data["image"][0]["width"]
    image_height = data["image"][0]["height"]
    features = data["features"]
    
    # image_size = (image_width, image_height) ######json 오류 없어지면 이걸로 대체하면 됌
    if os.path.basename(label_path)[2] == 'L':
        image_size = (512,512)
    elif os.path.basename(label_path)[2] == 'U':
        image_size = (1024,1024)
        
    class_mapping = {i: i for i in range(0,31)}
    # class_mapping[31] = 0
    # mat_image = create_mat(image_size, features, class_mapping)
    
    # Create an empty image with a white background
    label = Image.new("L", image_size, 255)
    draw = ImageDraw.Draw(label)
   
    cnt = 0
    feat_cnt = {}
    for feature in features:        
        feat_cnt[cnt]=len(feature['geometry']['coordinates'])        
        cnt +=1
    
    for key, value in feat_cnt.items():
        if value >= 2:
            for num in range(len(features[key]['geometry']['coordinates'])):    
                polygon = features[key]['geometry']['coordinates'][num]
                category_id = features[key]['properties']['categories_id']
                for sublist in polygon:
                    for i in range(len(sublist)):
                        sublist[i] = abs(sublist[i])
                polygon = [item for sublist in polygon for item in sublist]                        
                class_value = class_mapping.get(category_id, 0)  # Map class to pixel value
                draw.polygon(polygon, outline=class_value, fill=class_value)
                
    for key, value in feat_cnt.items():
        if value < 2:
            category_id = features[key]['properties']['categories_id']
            if features[key]['geometry']['coordinates']:
                polygon = features[key]['geometry']['coordinates'][0]
                for sublist in polygon:
                    for i in range(len(sublist)):
                        sublist[i] = abs(sublist[i])
                polygon = [item for sublist in polygon for item in sublist]        
                class_value = class_mapping.get(category_id, 0)  # Map class to pixel value
                draw.polygon(polygon, outline=class_value, fill=class_value)
                
    # if loc == "U":
    label = np.array(label)[sampling_coords[:,1], sampling_coords[:,0]].reshape(output_size,output_size)
        # selected_band_image = selected_band_image[sampling_coords[:,1], sampling_coords[:,0], ].reshape(512,512,120)
    
    label = np.array(label)    
    # label[label==30] = 255
    label[label==255] = 30
    # temp = label*8
    return label



def L_file(image_path, imgtype, sampling_coords, output_size): # land
    # imgtype = ["RA", "RE"]
    image_shape = (512,512)
    num_channels = 10
    # total_channel = 120
    total_channel = 100    
    band = 100
    
    file_dir, file_name = os.path.split(image_path)
    base_name, file_extension = os.path.splitext(file_name)
    
    concatenated_image = np.zeros(image_shape + (total_channel,), dtype=np.uint16)
    if total_channel == 100:
        for i in range(2, int(total_channel/num_channels) +2):
            tif_file_path = os.path.join(file_dir, f"{base_name}_{imgtype}{i:02d}{file_extension}")
            try:
                os.path.exists(tif_file_path)
                img = tifffile.imread(tif_file_path)
                concatenated_image[..., (i - 2) * num_channels:(i-1) * num_channels] = img
            except FileNotFoundError as e:
                raise e
    elif total_channel == 120:
        for i in range(1, int(total_channel/num_channels) +1):
            tif_file_path = os.path.join(file_dir, f"{base_name}_{imgtype}{i:02d}{file_extension}")
            try:
                os.path.exists(tif_file_path)
                img = tifffile.imread(tif_file_path)
                concatenated_image[..., (i - 1) * num_channels:i * num_channels] = img
            except FileNotFoundError as e:
                raise e
    concatenated_image = concatenated_image[sampling_coords[:,1], sampling_coords[:,0], ].reshape(output_size,output_size,band)
    return concatenated_image

def U_file(image_path, imgtype, sampling_coords, output_size): #under water
    # #120 band
    # selected_band_list = [10,12,13,15,17,19,20,22,24,25,27,29,31,32,34,36,37,39,41,42,44,46,48,49,51,53,54,56,58,59,61,63,65,66,68,70,71,73,75,76,78,80,81,83,85,87,88,90,92,93,95,97,98,100,102,104,105,107,108,110,112,114,115,117,118,120,122,124,125,127,129,130,132,134,135,137,139,140,142,144,145,147,149,150,152,154,155,157,158,160,162,164,165,167,169,170,172,174,175,177,179,180,182,183,185,187,188,190,192,193,195,197,198,200,202,203,205,207,208,209,]
    # band = 120
    
    # 100 band
    selected_band_list = [27,29,31,32,34,36,37,39,41,42,44,46,48,49,51,53,54,56,58,59,61,63,65,66,68,70,71,73,75,76,78,80,81,83,85,87,88,90,92,93,95,97,98,100,102,104,105,107,108,110,112,114,115,117,118,120,122,124,125,127,129,130,132,134,135,137,139,140,142,144,145,147,149,150,152,154,155,157,158,160,162,164,165,167,169,170,172,174,175,177,179,180,182,183,185,187,188,190,192,193,]
    band = 100
    image_shape = (1024,1024)
    num_channels = 14
    total_channel = 210
    
    file_dir, file_name = os.path.split(image_path)
    base_name, file_extension = os.path.splitext(file_name)
    
    concatenated_image = np.zeros(image_shape + (total_channel,), dtype=np.uint16)
    for i in range(1, int(total_channel/num_channels) +1):
        tif_file_path = os.path.join(file_dir, f"{base_name}_{imgtype}{i+20:02d}{file_extension}")
        try:
            os.path.exists(tif_file_path)
            img = tifffile.imread(tif_file_path)
            concatenated_image[..., (i - 1) * num_channels:i * num_channels] = img        
        except FileNotFoundError as e:
            raise e
            


    selected_band_image = concatenated_image[:,:,selected_band_list]
    ##############(1024->512)##########################################
    selected_band_image = selected_band_image[sampling_coords[:,1], sampling_coords[:,0], ].reshape(output_size,output_size,band)
    ##############(1024->512)##########################################
    
    
    return selected_band_image
    

def D_file(image_path, imgtype, sampling_coords, output_size): #drone
    image_shape = (256,256)
    num_channels = 37    
    selected_band_list = [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86]

    total_channel = 111 # RE/RA 36, 37, 38 (각 37개씩,,, 37x3 = 111)
    band = 80
    
    file_dir, file_name = os.path.split(image_path)
    base_name, file_extension = os.path.splitext(file_name)
    
    concatenated_image = np.zeros(image_shape + (total_channel,), dtype=np.uint16)
    for i in range(1, int(total_channel/num_channels) +1):
        tif_file_path = os.path.join(file_dir, f"{base_name}_{imgtype}{i+35:02d}{file_extension}")
        try:
            os.path.exists(tif_file_path)
            img = tifffile.imread(tif_file_path)
            concatenated_image[..., (i - 1) * num_channels:i * num_channels] = img        
        except FileNotFoundError as e:
            raise e
            


    selected_band_image = concatenated_image[:,:,selected_band_list]
    ##############(256->?)##########################################
    selected_band_image = selected_band_image[sampling_coords[:,1], sampling_coords[:,0], ].reshape(output_size,output_size,band)
    ##############(256->?)##########################################    
    
    return selected_band_image

def sampling_point(image_size, y):
    # 이미지 크기
    image_width = image_size
    image_height = image_size

    # y 변수 설정
    # y = 512  # y 값을 변경하여 원하는 등분 수를 얻을 수 있습니다.

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
    output_size = 512    
    output_size_drone = 256
    sampling_coords = [sampling_point(512, output_size), sampling_point(1024, output_size), sampling_point(256, output_size_drone)]
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "w") as log:
        log.write(f"Start Time [{formatted_time}]\n")
    
    for root, dirs, files in os.walk(image_folder):
        # if "02.수중 및 지상 초분광" in root:
            for image_filename in files:
                if image_filename.endswith('.tif'):
                    cnt +=1
    
    total_cnt = cnt
    cnt= 0
    error_cnt=0
    
    for root, dirs, files in os.walk(image_folder):
        # if "02.수중 및 지상 초분광" in root:
            for image_filename in files:
                if image_filename.endswith('.tif'):
                    base_name, file_extension = os.path.splitext(image_filename)
                    ####################################################################################                    
                    prefix = '_'.join(image_filename.split('_')[:-1])  # Extract the common prefix

                    if prefix not in file_paths_dict:
                        file_paths_dict[prefix] = []
                    else:
                        continue
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
                            mat_image_RA = L_file(image_path, imgtype[0], sampling_coords[0], output_size)
                            mat_image_RE = L_file(image_path, imgtype[1], sampling_coords[0], output_size)                            
                            label_path = label_path.replace(".tif", "_RE01.json")                            
                            mat_label = create_label_mat(label_path, prefix[2], sampling_coords[0], output_size)
                        elif prefix[2] == 'U':                        
                            print(f"{cnt}/{int(total_cnt/30)}({total_cnt})\t{image_path}\n") 
                            mat_image_RA = U_file(image_path, imgtype[0], sampling_coords[1], output_size)
                            mat_image_RE = U_file(image_path, imgtype[1], sampling_coords[1], output_size)
                            label_path = label_path.replace(".tif", "_RE21.json")
                            mat_label = create_label_mat(label_path, prefix[2], sampling_coords[1], output_size)                            
                        elif prefix[2] == 'D':
                            mat_image_RA = D_file(image_path, imgtype[0], sampling_coords[2], output_size_drone)
                            mat_image_RE = D_file(image_path, imgtype[1], sampling_coords[2], output_size_drone)
                            label_path = label_path.replace(".tif", "_RE36.json")
                            mat_label=create_label_mat(label_path, prefix[2], sampling_coords[2], output_size_drone)
                                                  
                        # img_RA = (np.clip(np.array(mat_image_RA)[:,:,[15,39,80]].astype(np.float32)/40000, 0.0, 1.0)*255).astype(np.uint8)
                        # img_RE = (np.clip(np.array(mat_image_RE)[:,:,[15,39,80]].astype(np.float32)/40000, 0.0, 1.0)*255).astype(np.uint8)

                        io.savemat(mat_path_RA, {'image': np.array(mat_image_RA), 'label' : mat_label})
                        io.savemat(mat_path_RE, {'image': np.array(mat_image_RE), 'label' : mat_label})
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