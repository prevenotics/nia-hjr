import os
import csv
import yaml
import random
from scipy.io import loadmat
from collections import Counter
import os
from multiprocessing import Pool
import datetime
import pickle


def process_mat_file(file_path):
    mat = loadmat(file_path)
    label = mat['label']
    label_flat = label.flatten()
    return label_flat

if __name__ == "__main__":
    with open('/root/work/hjr/IEEE_TGRS_SpectralFormer/config.yaml') as f:
        cfg = yaml.safe_load(f)
    dataset_path= cfg['path']['dataset_path']    
    list_mat = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.mat'):
                file_path = os.path.join(root, file)
                list_mat.append(file_path)

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")    
    print(f"Start Time [{formatted_time}]\n")
    
    
    
    multi_num = 32
    # 병렬 처리를 위한 프로세스 풀 생성
    with Pool(processes=multi_num) as pool:  # 여기서 4는 병렬로 처리할 프로세스 수입니다
        results = pool.map(process_mat_file, list_mat)

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(len(list_mat))
    print(f"End Time [{formatted_time}]\n")
    
    # 결과를 합침
    value_counts = Counter()
    for result in results:
        value_counts.update(result)

    sorted_counts = sorted(value_counts.items(), key=lambda item: item[0])
    
    serialized_counts = dict(sorted_counts)
    with open(f'value_counts_{multi_num}.pkl', 'wb') as file:
        pickle.dump(serialized_counts, file)
    # with open('value_counts.pkl', 'rb') as file:
    #     loaded_counts = pickle.load(file)
    
    for value, count in value_counts.items():
        print(f"{value}: {count}개")