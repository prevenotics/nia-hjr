from collections import defaultdict
import os
import math
import pandas as pd
import random

os.chdir('/root/work/hjr/nia-hjr/preprocessing')

folder_counts = {
    '01.갈파래류': 80,
    '02.청각류': 24,
    '03.대마디말류': 20,
    '04.그물바탕말류': 9,
    '05.모자반류': 397,
    '06.나래미역류': 100,
    '07.감태류': 183,
    '08.유절산호말류': 100,
    '09.무절산호말류': 161,
    '10.우뭇가사리류': 120,
    '11.도박류': 27,
    '12.돌가사리류': 19,
    '13.새우말류': 31,
    '14.거머리말류': 88,
    '15.암반류': 32,
    '16.모래류': 578,
    '17.인공어초류': 40,
    '18.성게류': 34,
    '19.불가사리류': 35,
    '20.소라류': 343,
    '21.군소_전복류': 168,
    '22.해면류': 16,
    '23.담치류': 64,
    '24.따개비류': 28,
    '25.고둥류': 140,
    '26.군부류': 54,
    '27.조개류': 55,
    '28.연성_경성산호류': 72,
    '29.해양쓰레기류': 113,
    '30.폐어구류': 46,
    '31.기타': 16
}

folder_counts_drone = {
    '15.암반류': 197,
    '16.모래류': 122
}
OA_thres = 0.2
imgtype='RE'
data_file = f'LU_{imgtype}.txt'  # 주어진 파일명
out_file = f'LU_{imgtype}_selected.txt'
# data_file = f'D_{imgtype}.txt'  # 주어진 파일명
# out_file = f'D_{imgtype}_selected.txt'


# 각 클래스별로 선택한 파일 경로를 담을 딕셔너리 초기화
selected_paths = {key: [] for key in folder_counts}


with open(data_file, 'r') as file:
    lines = file.readlines()

random.shuffle(lines)

# with open(data_file, 'r') as file:
for line in lines:

    # 주어진 파일의 각 줄에서 첫 번째 숫자 값이 0.7을 넘는지 확인하여 선택
    parts = line.split('\t')
    path = parts[0]
    class_name = path.split('/')[6]  # 클래스 이름 추출 (경로에 따라 조정 필요)
    value = float(parts[2])
    
    if value > OA_thres and class_name in folder_counts:
        # 선택한 클래스별 개수에 맞게 파일 경로 선택
        if len(selected_paths[class_name]) < folder_counts[class_name]:
            selected_paths[class_name].append(path)

# 선택된 파일 경로를 출력
with open(out_file, 'w') as output_file:
    print('start')
for class_name, paths in selected_paths.items():    
    with open(out_file, 'a') as output_file:
        for path in paths:
            output_file.write(f"{path}\n")    
    shortage = folder_counts[class_name] - len(paths)
    print(f"Class: {class_name}, Selected: {len(paths)}, Required: {folder_counts[class_name]}, Shortage: {shortage}")