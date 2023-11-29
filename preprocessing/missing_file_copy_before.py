import os 
os.chdir('/root/work/hjr/nia-hjr/preprocessing')

# RA_list.txt와 RE_list.txt의 내용을 읽어옵니다.
with open('total_mat_RA_selected.txt', 'r') as file:
    ra_list = file.readlines()

with open('total_mat_RE_selected.txt', 'r') as file:
    re_list = file.readlines()


ra_filenames = [line.split('/')[-1].strip()[:-7] for line in ra_list]
re_filenames = [line.split('/')[-1].strip()[:-7] for line in re_list]


ra_common = [prefix+'_RA.mat' for prefix in ra_filenames if prefix in re_filenames]
ra_missing = [prefix+'_RA.mat' for prefix in re_filenames if prefix not in ra_filenames]

# 결과를 파일로 저장합니다.
with open('RA_list_common.txt', 'w') as file:
    file.write('\n'.join(ra_common))

with open('RA_list_missing.txt', 'w') as file:
    file.write('\n'.join(ra_missing))