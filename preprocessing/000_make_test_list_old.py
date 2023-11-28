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

def each_class_avg():
    # 파일에서 클래스별 OA와 kappa 값을 저장할 defaultdict 생성
    class_stats = defaultdict(lambda: {"count": 0, "OA_sum": 0, "kappa_sum": 0})

    # 파일에서 데이터를 읽어옴 (여기서는 파일을 줄 단위로 읽어오는 가정)
    with open('LU_RE_out.txt', 'r') as file:
        file.readline()
        for line in file:
            # 각 줄을 탭으로 분리하여 필요한 정보를 추출
            parts = line.strip().split('\t')
            
            # 클래스 이름 추출 (여기서는 파일 경로에서 추출하도록 가정)
            class_name = parts[0].split('/')[6]  # 파일 경로에서 클래스 이름 추출 (여러 경우에 따라 수정 필요)
            
            # OA와 kappa 값 추출
            # oa_value = float(parts[2].split()[1])
            if len(parts) != 1:
                oa_value = float(parts[1])
                # kappa_value = float(parts[4].split()[1])
                # kappa_value = float(parts[2])
            
                # 해당 클래스의 값에 현재 값 더하고 개수 1 증가
                if not math.isnan(oa_value):
                    class_stats[class_name]["OA_sum"] += oa_value
                    # class_stats[class_name]["kappa_sum"] += kappa_value
                    class_stats[class_name]["count"] += 1

    # 클래스별로 OA와 kappa의 평균값 계산
    for class_name, values in class_stats.items():
        count = values["count"]
        oa_avg = values["OA_sum"] / count if count > 0 else 0  # OA의 평균
        # kappa_avg = values["kappa_sum"] / count if count > 0 else 0  # kappa의 평균
        print(f"Class: {class_name}, Count: {count}, OA Avg: {oa_avg:.3f}")

##############################################################################################################################
# # 파일 읽기
# file_path = 'LU_RE.txt'  # 파일 경로에 맞게 수정해주세요
# output_file_path = 'top_20_percent_excluding_top_10.txt'
# column_names = ['file_path', 'OA', 'kappa']  # 열 이름에 맞게 수정해주세요

# # data = pd.read_csv(file_path, sep='\t', names=column_names)
# data = pd.read_csv(file_path, sep='\t', names=column_names, dtype={'file_path': str})


# # 클래스별로 그룹화하여 개수, 평균 등 계산하기
# class_counts = data['file_path'].apply(lambda x: x.split('/')[6]).value_counts()
# class_means = data.groupby(data['file_path'].apply(lambda x: x.split('/')[6])).mean()
# classes_over_100 = class_counts[class_counts > 100].index.tolist()

# # OA 상위 20%~10%에 해당되는 데이터 추출
# class_oa_sorted = data.groupby(data['file_path'].apply(lambda x: x.split('/')[6])).mean()['OA'].sort_values(ascending=False)

# # 상위 20%에 해당되는 데이터
# top_20_percent = {}
# # for class_name, mean_oa in class_oa_sorted.items():
# #     class_data = data[data['file_path'].apply(lambda x: x.split('/')[6]) == class_name]
# #     top_20 = class_data[class_data['OA'] >= mean_oa]
# #     top_20_percent[class_name] = top_20
# for class_name, group_data in data.groupby(data['file_path'].apply(lambda x: x.split('/')[6])):
#     group_data_sorted = group_data.sort_values(by='OA', ascending=False)
#     top_20_index = int(len(group_data_sorted) * 0.2)
#     top_20_data = group_data_sorted.iloc[:top_20_index]
#     top_20_percent[class_name] = top_20_data
    
# top_20_percent_excluding_top_10 = {}
# for class_name, data_frame in top_20_percent.items():
#     total_count = len(data_frame)
#     top_10_cutoff = int(total_count * 0.5)  # 상위 10%에 해당하는 개수 계산
#     top_20_percent_excluding_top_10[class_name] = data_frame[:-top_10_cutoff]  


# existing_data = pd.concat(top_20_percent_excluding_top_10.values())



# max_samples_per_class = 20  # 각 클래스당 최대 선택 가능한 샘플 수
# total_samples = 0
# additional_data = []

# while total_samples < 3000:
#     classes_over_100 = existing_data['file_path'].apply(lambda x: x.split('/')[6]).value_counts()
#     classes_lacking = classes_over_100[classes_over_100 < 100].index.tolist()
    
#     if not classes_lacking:
#         break
    
#     selected_class = random.choice(classes_lacking)
    
#     class_data = data[data['file_path'].apply(lambda x: x.split('/')[6]) == selected_class]
#     high_oa_data = class_data[class_data['OA'] >= 0.7]
    
#     if len(high_oa_data) > 0:
#         sample_data = high_oa_data.sample(n=min(max_samples_per_class, len(high_oa_data)), replace=False)
        
#         for idx, row in sample_data.iterrows():
#             if total_samples >= 3000:
#                 break
            
#             # 중복 확인
#             if not any((additional_data == row).all(1)):
#                 additional_data.append(row)
#                 total_samples += 1

# additional_data_df = pd.DataFrame(additional_data)
# final_data = pd.concat([existing_data, additional_data_df])

# # 결과를 파일로 저장
# final_data.to_csv(output_file_path, sep='\t', index=False)
##############################################################################################################################
# additional_data = []
# while len(additional_data) < 3000:
#     classes_over_100 = existing_data['file_path'].apply(lambda x: x.split('/')[6]).value_counts()
#     classes_lacking = classes_over_100[classes_over_100 < 100].index.tolist()
    
#     if not classes_lacking:
#         break
    
#     selected_class = random.choice(classes_lacking)
#     class_data = data[data['file_path'].apply(lambda x: x.split('/')[6]) == selected_class]
#     high_oa_data = class_data[class_data['OA'] >= 0.7]
#     if len(high_oa_data) > 0:
#         sample_data = high_oa_data.sample(n=min(3000 - len(additional_data), len(high_oa_data)))
#         additional_data.append(sample_data)

# additional_data_df = pd.concat(additional_data)
# final_data = pd.concat([existing_data, additional_data_df])

# # 결과를 파일로 저장
# final_data.to_csv(output_file_path, sep='\t', index=False)

##############################################################################################################################





# 데이터를 읽고 파일에서 DataFrame을 만듭니다.
file_path = 'LU_RE.txt'  # 파일 경로를 수정해주세요
output_file_path = 'LU_RE_out.txt'
column_names = ['file_path', 'OA', 'kappa']  # 열 이름을 수정해주세요
data = pd.read_csv(file_path, sep='\t', names=column_names, dtype={'file_path': str})

class_counts = data['file_path'].apply(lambda x: x.split('/')[6]).value_counts()
total_counts = len(data)
class_oa_sorted = data.groupby(data['file_path'].apply(lambda x: x.split('/')[6])).mean()['OA'].sort_values(ascending=False)
top_20_percent = {}

for class_name, mean_oa in class_oa_sorted.items():
    class_data = data[data['file_path'].apply(lambda x: x.split('/')[6]) == class_name]
    class_size = class_counts[class_name]
    top_20_count = int(class_size * 0.2)  # 상위 20%에 해당하는 개수 계산
    top_20 = class_data[:top_20_count]
    top_20_percent[class_name] = top_20

top_20_percent_excluding_top_10 = {}
for class_name, data_frame in top_20_percent.items():
    total_count = len(data_frame)
    top_10_cutoff = int(total_count * 0.5)  # 상위 10%에 해당하는 개수 계산
    top_20_percent_excluding_top_10[class_name] = data_frame[:-top_10_cutoff]


# 추가 데이터가 필요한 클래스 중 100개 이상인 클래스를 선택합니다.
classes_over_100 = {class_name: df for class_name, df in top_20_percent_excluding_top_10.items() if len(df) > 100}
additional_data = []
while len(additional_data) < 3000 - sum(len(df) for df in top_20_percent_excluding_top_10.values()):
    selected_class = random.choice(list(classes_over_100.keys()))
    class_data = data[data['file_path'].apply(lambda x: x.split('/')[6]) == selected_class]
    high_oa_data = class_data[class_data['OA'] >= 0.7]
    if len(high_oa_data) > 0:
        sample_data = high_oa_data.sample(n=min(3000 - len(additional_data), len(high_oa_data)))
        additional_data.append(sample_data)

# 최종 리스트를 합쳐 최종 3000개를 만듭니다.
final_data = []
final_data.extend([df for df in top_20_percent_excluding_top_10.values()])
final_data.extend(additional_data)

# 최종 데이터프레임으로 변환합니다.
final_df = pd.concat(final_data)
final_df.to_csv(output_file_path, sep='\t', index=False)

each_class_avg()

