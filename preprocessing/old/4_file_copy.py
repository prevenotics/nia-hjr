import shutil
import os


with open('output_common_files.txt', 'r') as file:
    file_list = file.read().splitlines()



cnt=0
for file in file_list:
    source_image_dir= os.path.dirname(file)
    source_label_dir = source_image_dir.replace('1.원천데이터', '2.라벨링데이터')
    image_destination_folder = os.path.dirname(file).replace('X:\\NIA\\Final_Result\\', 'D:\\DATA\\hjr\\dataset\\')
    label_destination_folder = os.path.dirname(file).replace('X:\\NIA\\Final_Result\\', 'D:\\DATA\\hjr\\dataset\\').replace('1.원천데이터', '2.라벨링데이터')
    file_name, file_extension = os.path.splitext(os.path.basename(file))    
    for i in range(1,21):
        image_new_file_name = f"{file_name}_RA{i:02}{file_extension}"
        label_new_file_name = f"{file_name}_RA{i:02}.json"
        if not os.path.exists(image_destination_folder):
            os.makedirs(image_destination_folder)
        if not os.path.exists(label_destination_folder):
            os.makedirs(label_destination_folder)
        shutil.copy(os.path.join(source_image_dir,image_new_file_name), os.path.join(image_destination_folder, image_new_file_name))
        shutil.copy(os.path.join(source_label_dir,label_new_file_name), os.path.join(label_destination_folder, label_new_file_name))
    for i in range(1,21):
        image_new_file_name = f"{file_name}_RE{i:02}{file_extension}"
        label_new_file_name = f"{file_name}_RE{i:02}.json"
        if not os.path.exists(image_destination_folder):
            os.makedirs(image_destination_folder)
        if not os.path.exists(label_destination_folder):
            os.makedirs(label_destination_folder)
        shutil.copy(os.path.join(source_image_dir,image_new_file_name), os.path.join(image_destination_folder, image_new_file_name))
        shutil.copy(os.path.join(source_label_dir,label_new_file_name), os.path.join(label_destination_folder, label_new_file_name))
    print(f'{cnt}')
    cnt+=1
    

