import os
import json
import numpy as np
# import rasterio
# from rasterio.transform import from_origin
from PIL import Image, ImageDraw
from scipy import io

# json_folder = r'D:/DATA/hjr/dataset/2.라벨링데이터/'
# output_parent_folder = r'D:/DATA/hjr/dataset/2.라벨링데이터_mat/'
json_folder = r'D:/DATA/hjr/dataset/2.라벨링데이터/'
output_parent_folder = r'D:/DATA/hjr/dataset/temp/'
# Create the TIFF folder if it doesn't exist
# if not os.path.exists(tiff_folder):
#     os.makedirs(tiff_folder)

width = 512  # 넓이
height = 512  # 높이
num_categories = 30  # 카테고리 수
background_color = 0
category_colors = np.linspace(0, 255, num_categories, dtype=np.uint8)

def create_mat(image_size, features, class_mapping):
    # Create an empty image with a white background
    image = Image.new("L", image_size, 255)
    draw = ImageDraw.Draw(image)
   
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


    
    return image

def main():        
    cnt = 0
    for root, dirs, files in os.walk(json_folder):
        # if "001.RGB" in root:  # 경로에 "001.RGB"가 포함되어 있는지 확인
            for json_filename in files:
                if json_filename.endswith('.json'):
                    json_path = os.path.join(root, json_filename)
                    
                    relative_path = os.path.relpath(root, json_folder)
                    tiff_folder = os.path.join(output_parent_folder, relative_path)
                    os.makedirs(tiff_folder, exist_ok=True) 
                    
                    print(f"Currnet file is {json_path}")
                    with open(json_path, "r") as json_file:
                        data = json.load(json_file)
                    
                    image_id = data["image"][0]["id"]
                    image_width = data["image"][0]["width"]
                    image_height = data["image"][0]["height"]
                    features = data["features"]
                    
                    image_size = (image_width, image_height)
                    # category_values = np.linspace(0, 255, num=len(data['features']), dtype=np.uint8)
                    # category_values = np.linspace(0, 255, num=num_categories, dtype=np.uint8)
                    
                    
                    # Create a dictionary to map class IDs to pixel values
                    # class_mapping = {category["id"]: i for i, category in enumerate(data["categories"], start=1)}
                    class_mapping = {i: i * 8 for i in range(0,31)}
                    # class_mapping[31] = 0
                    
                    # Create TIFF image
                    tiff_image = create_mat(image_size, features, class_mapping)
                    
                    class_mapping = {i: i for i in range(0,31)}
                    # class_mapping[31] = 0
                    mat_image = create_mat(image_size, features, class_mapping)
                    
                    # Save the TIFF image
                    tiff_filename = os.path.splitext(json_filename)[0] + ".tif"
                    tiff_path = os.path.join(tiff_folder, tiff_filename)
                    tiff_image.save(tiff_path)
                    
                    mat_filename = os.path.splitext(json_filename)[0] + ".mat"                    
                    mat_path = os.path.join(tiff_folder, mat_filename)                    
                    io.savemat(mat_path, {'label' : np.array(mat_image)})
                    
                    # print(f"Converted {json_filename} to {tiff_filename}")
                    cnt +=1
    print(f"Total count =  {cnt}")
                    
    
if __name__ == "__main__":
    main()