import os
image_folder = r'/root/work/hjr/dataset/3.mat_mult_OutSize256_error_cnt/'
output_folder = image_folder.replace("3.mat_mult_OutSize256_error_cnt/","4.mat_mult_OutSize256_error_cnt/")

class_counts = {str(i): 0 for i in range(31)}
for root, dirs, files in os.walk(output_folder):
    for file_name in files:
        if file_name.endswith('.mat'):
            # 파일 경로에서 클래스 번호 추출
            file_name_only, _ = os.path.splitext(file_name)
            parts = file_name_only.split('_')
            class_num = parts[2]  # 클래스 번호는 파일 이름에서 두 번째 부분
            class_counts[class_num] += 1
            
            
for class_num, count in class_counts.items():
    print(f'Class {class_num}: {count} files')