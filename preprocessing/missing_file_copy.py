import os
import shutil
os.chdir('/root/work/hjr/nia-hjr/preprocessing')
# 파일명이 있는 텍스트 파일 경로
file_list_txt = 'RA_list_common.txt'

# 복사할 디렉토리 및 복사 실패 목록을 저장할 파일 경로
success_file = 'mssing_file_copy_success.txt'
fail_file = 'mssing_file_copy_fail.txt'

# 파일명 리스트 읽기
# file_list = []
# with open(file_list_txt, 'r') as file:
#     file_list = file.readlines()
# file_list = [f.strip() for f in file_list]

file_list = []
with open(file_list_txt, 'r') as file:
    for line in file:
        # 파일명만 추출 (예상되는 형식에 따라 적절히 경로 부분을 제거해야 함)
        # filename = line.split('/')[-1].strip()  # '/'를 기준으로 경로를 나눈 후 마지막 요소인 파일명만 추출
        # file_list.append(filename)
        file_list.append(line.strip())

# 파일 찾아서 복사하기
success_count = 0
fail_count = 0
with open(fail_file, 'w') as fail_txt:
    print("")
    
for filename in file_list:
    
    root_dir = '/workspace/dataset/3.mat'
    
    # 파일을 찾음
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            source_path = os.path.join(root, filename)            
            destination_path = source_path.replace('3.mat', '3.mat_temp_copy')
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy(source_path, destination_path)
            success_count += 1
            break
    else:
        
        with open(fail_file, 'a') as fail_txt:
            fail_txt.write(f'File not found: {filename}\n')
        fail_count += 1

# 복사 성공한 파일 개수와 실패한 파일 목록을 기록
with open(success_file, 'w') as success_txt:
    success_txt.write(f'Successfully copied {success_count} files.\n')
    success_txt.write(f'Failed to find {fail_count} files.\n')
