import csv
import os
os.chdir('/root/work/hjr/nia-hjr/preprocessing')


input_csv = '../../dataset/test_final_RE.csv'  # 여러 파일 경로가 있는 CSV 파일명을 넣어주세요
output_csv = '../../dataset/test_final_RE_filtered.csv' 

existing_files = []
with open(input_csv, 'r', newline='') as infile:
    reader = csv.reader(infile)
    rows = list(reader)
    for row in rows:
        file_path = row[0]  # 파일 경로가 있는 열의 인덱스로 변경해주세요
        if os.path.exists(file_path):
            existing_files.append(row)

# 필터링된 파일 경로를 가지고 새로운 CSV 파일 생성
with open(output_csv, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(existing_files)