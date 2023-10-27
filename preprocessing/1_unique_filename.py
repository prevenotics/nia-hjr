import os
from collections import Counter
import unicodedata

# 전체 데이터셋 image/label 폴더에서 dir /b/s >list_image.txt, dir /b/s >list_label.txt 하여 생성된 파일을 기준으로 RE, RA를 제거한 unique 파일 목록 생성
# 현재 데이터셋 개수 확인(unique 데이터 기준)

folder_counts = {
    '01.갈파래류': 0,
    '02.청각류': 0,
    '03.대마디말류': 0,
    '04.그물바탕말류': 0,
    '05.모자반류': 0,
    '06.나래미역류': 0,
    '07.감태류': 0,
    '08.유절산호말류': 0,
    '09.무절산호말류': 0,
    '10.우뭇가사리류': 0,
    '11.도박류': 0,
    '12.돌가사리류': 0,
    '13.새우말류': 0,
    '14.거머리말류': 0,
    '15.암반류': 0,
    '16.모래류': 0,
    '17.인공어초류': 0,
    '18.성게류': 0,
    '19.불가사리류': 0,
    '20.소라류': 0,
    '21.군소_전복류': 0,
    '22.해면류': 0,
    '23.담치류': 0,
    '24.따개비류': 0,
    '25.고둥류': 0,
    '26.군부류': 0,
    '27.조개류': 0,
    '28.연성_경성산호류': 0,
    '29.해양쓰레기류': 0,
    '30.폐어구류': 0,
    '31.기타': 0
}

image_or_label = 'image'
image_or_label = 'label'

#window
with open(f'list_{image_or_label}_231026_re.txt', 'r') as file:
    file_paths = file.readlines()
# #linux
# with open(f'list_{image_or_label}_231026_re.txt', 'r', encoding='UTF8') as file:
#     file_paths = file.readlines()

# Create a dictionary to group file paths by their common prefix
file_paths_dict = {}
for file_path in file_paths:
    file_extension = os.path.splitext(file_path)[1]
    prefix = '_'.join(file_path.split('_')[:-1])  # Extract the common prefix
    if prefix not in file_paths_dict:
        file_paths_dict[prefix] = []
    file_paths_dict[prefix].append(file_path)

# Create a list of modified file paths with the common prefix
modified_file_paths = [prefix + file_extension for prefix in file_paths_dict.keys()]

modified_file_paths.sort()

# folder_counts = Counter(path.split("\\")[4] for path in modified_file_paths)
for path in modified_file_paths:
    folder_name = path.split("\\")[4]
    # folder_name = path.split("/")[10]
    if folder_name in folder_counts:
        folder_counts[folder_name] +=1
        

# Write the modified file paths to a new output file
# with open('output_label.txt', 'w') as file:    
#     file.writelines(line + '\n' for line in modified_file_paths)
with open(f'output_{image_or_label}.txt', 'w') as file:
    file.writelines(modified_file_paths)
    
with open(f'output_{image_or_label}_stat.txt', 'w') as file:    
    max_key_length = max(len(key) for key in folder_counts.keys())
    for key, value in folder_counts.items():
        file.write(f"{key.ljust(max_key_length)}: {value}\n")
