import os

# 

# 두 개의 파일에서 파일 경로를 포함한 목록을 읽어옴
with open('output_image.txt', 'r') as file1:
    file_list1 = file1.read().splitlines()

with open('output_label.txt', 'r') as file2:
    file_list2 = file2.read().splitlines()

# 파일명만 추출하여 집합(set)에 저장
file_names1 = {os.path.splitext(os.path.basename(file))[0] for file in file_list1}
file_names2 = {os.path.splitext(os.path.basename(file))[0] for file in file_list2}

# 겹치는 파일명 찾기
common_file_names = file_names1 & file_names2

# 겹치는 파일들의 경로와 파일명을 저장할 리스트
common_files = []

# 겹치는 파일의 경로와 파일명 추출
for file in file_list1:
    if os.path.splitext(os.path.basename(file))[0] in common_file_names:
        common_files.append(file)


# 겹치는 파일 경로와 파일명을 따로 저장할 파일
with open('output_common_files.txt', 'w') as common_file:
    for common_file_info in common_files:
        common_file.write(common_file_info + '\n')

print("겹치는 파일 목록이 common_files.txt 파일에 저장되었습니다.")
