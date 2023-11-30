# import os
# from scipy import io
# import numpy as np

# image_folder = r'/workspace/dataset/3.mat_mult_OutSize256_error_cnt'
# output_folder = image_folder.replace("3.mat_mult_OutSize256_error_cnt/","4.mat_mult_OutSize256_error_cnt/")


# for root, dirs, files in os.walk(output_folder):
#     for file_name in files:
#         if file_name.endswith('.mat'):
#             image_path = os.path.join(root, file_name)                    
#             # os.makedirs(output_folder, exist_ok=True)
            
#             mat_file = io.loadmat(image_path)
#             # image = mat_file['image']
#             label = mat_file['label']
            
#             indices = np.where(label == 255)
            
#             if indices[0].size > 0:
#                 print(f"{image_path}")



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


image_folder = r'/workspace/dataset//1.원천데이터/'
# json_folder = r'D:/DATA/hjr/dataset/2.라벨링데이터/'
# output_image_folder = r'D:/DATA/hjr/dataset/1.image_mat/'
# output_json_folder = r'D:/DATA/hjr/dataset/2.label_mat/'
# output_mat_folder = r'D:/DATA/hjr/dataset/3.mat/'

# input_path= input("input path : ")
# image_folder = input_path

output_mat_folder = image_folder.replace("1.원천데이터","3.mat_temp")

def create_label_mat(label_path, loc, sampling_coords):
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
            polygon = features[key]['geometry']['coordinates'][0]
            for sublist in polygon:
                for i in range(len(sublist)):
                    sublist[i] = abs(sublist[i])
            polygon = [item for sublist in polygon for item in sublist]        
            class_value = class_mapping.get(category_id, 0)  # Map class to pixel value
            draw.polygon(polygon, outline=class_value, fill=class_value)
                
    if loc == "U":
        label = np.array(label)[sampling_coords[:,1], sampling_coords[:,0]].reshape(512,512)
        # selected_band_image = selected_band_image[sampling_coords[:,1], sampling_coords[:,0], ].reshape(512,512,120)
    return label


CLIPPING = 40000
def L_file(image_path, imgtype): # land
    # imgtype = ["RA", "RE"]
    image_shape = (512,512)
    num_channels = 10
    # total_channel = 120
    total_channel = 100
    
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
    
    return concatenated_image

def U_file(image_path, imgtype, sampling_coords): #under water
    #120 band
    selected_band_list = [10,12,13,15,17,19,20,22,24,25,27,29,31,32,34,36,37,39,41,42,44,46,48,49,51,53,54,56,58,59,61,63,65,66,68,70,71,73,75,76,78,80,81,83,85,87,88,90,92,93,95,97,98,100,102,104,105,107,108,110,112,114,115,117,118,120,122,124,125,127,129,130,132,134,135,137,139,140,142,144,145,147,149,150,152,154,155,157,158,160,162,164,165,167,169,170,172,174,175,177,179,180,182,183,185,187,188,190,192,193,195,197,198,200,202,203,205,207,208,209,]
    band = 120
    #100 band
    # selected_band_list = [27,29,31,32,34,36,37,39,41,42,44,46,48,49,51,53,54,56,58,59,61,63,65,66,68,70,71,73,75,76,78,80,81,83,85,87,88,90,92,93,95,97,98,100,102,104,105,107,108,110,112,114,115,117,118,120,122,124,125,127,129,130,132,134,135,137,139,140,142,144,145,147,149,150,152,154,155,157,158,160,162,164,165,167,169,170,172,174,175,177,179,180,182,183,185,187,188,190,192,193,]
    # band = 100
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
    selected_band_image = selected_band_image[sampling_coords[:,1], sampling_coords[:,0], ].reshape(512,512,band)
    ##############(1024->512)##########################################
    
    
    return selected_band_image
    

def D_file(image_path): #drone
    return 0

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
                if image_filename.endswith('.tif'):
                    cnt +=1
    
    total_cnt = cnt
    cnt= 0
    error_cnt=0
    
    # for root, dirs, files in os.walk(image_folder):
    #     # if "02.수중 및 지상 초분광" in root:
    #         for image_filename in files:
    #             if image_filename.endswith('.tif'):
                    # base_name, file_extension = os.path.splitext(image_filename)
                    # ####################################################################################                    
                    # prefix = '_'.join(image_filename.split('_')[:-1])  # Extract the common prefix

                    # if prefix not in file_paths_dict:
                    #     file_paths_dict[prefix] = []
                    # else:
                    #     continue
                    # ####################################################################################
                    
    image_path ='/workspace/dataset/2.라벨링데이터/11.도박류/02.수중 및 지상 초분광/002.TIFF/01L_A000_0045_20230922_1527_W01_RE01.json'
                        
    # image_path = os.path.join(root, f"{prefix}{file_extension}")
    label_path = image_path.replace("1.원천데이터","2.라벨링데이터")
    relative_path = os.path.relpath(root, image_folder)
    
    
    mat_label=create_label_mat(label_path, 'L', sampling_coords)

    io.savemat('/workspace/dataset/01L_A000_0045_20230922_1527_W01_RE_label.mat', {'label':np.array(mat_label)})

    # io.savemat(mat_path_RA, {'image': np.array(mat_image_RA), 'label' : np.array(mat_label)})
    # io.savemat(mat_path_RE, {'image': np.array(mat_image_RE), 'label' : np.array(mat_label)})

        
    
if __name__ == "__main__":
    main()