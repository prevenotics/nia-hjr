import shutil
import os

os.chdir('/root/work/hjr/nia-hjr/batch')
# 파일 목록이 있는 텍스트 파일 경로
file_list_path = 'temp_list.txt'

# 파일을 복사할 폴더와 복사된 파일을 저장할 폴더
source_folder = '4.mat_selected_100p'
destination_folder = '4.mat_selected_100p_50'
os.makedirs('/workspace/dataset/4.mat_selected_100p_50/', exist_ok=True)

# 텍스트 파일에서 파일 경로를 읽어와 복사
with open(file_list_path, 'r') as file:
    for line in file:
        # 각 줄의 줄바꿈 문자 제거 후 파일 경로 가져오기
        file_path = line.strip()
        
        # 파일 경로 변경
        # 예를 들어, source_folder/foo/bar/file.txt를 destination_folder/foo/bar/file.txt로 바꿈
        new_file_path = file_path.replace(source_folder, destination_folder)
        
        
        # 파일 복사
        shutil.copy2(file_path, new_file_path)  # copy2는 파일 속성까지 복사합니다.
        print(f"Copied '{file_path}' to '{new_file_path}'")
