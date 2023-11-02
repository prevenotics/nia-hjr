import os
import csv
import random
from xml.sax.handler import DTDHandler


# 이미지와 label이 매치되는 데이터들 기준으로 학습을 위한 csv 파일 생성

# os.chdir("/root/work/prevenotics/src/IEEE_TGRS_SpectralFormer")
# RA_or_RE = "RA"

# read_train = open("/root/work/dataset/data100_last/city/"+city_or_forest+name+".txt", "r").readlines()
read_train = open("output_common_files.txt", "r").readlines()
random.shuffle(read_train)


ratio = 10
index = 0

# with open("train_"+RA_or_RE+".csv", mode="w", newline="") as train_file, open("val_"+RA_or_RE+".csv", mode="w", newline="") as val_file, open("test_"+RA_or_RE+".csv", mode="w", newline="") as test_file:
with open("train.csv", mode="w", newline="") as train_file, open("val.csv", mode="w", newline="") as val_file, open("test.csv", mode="w", newline="") as test_file:
    for line in read_train:        
        image = line.replace("\n", "")        
        # label = image.replace("1.원천데이터", "2.라벨링데이터").replace("tif", "json")
        label = image.replace("1.원천데이터", "2.라벨링데이터_mat").replace("tif", "mat")
        data = [image, label]
        index+=1
        if index%ratio == 0:
            writer = csv.writer(val_file)
            writer.writerow(data)
        elif index%ratio == 1:
            writer = csv.writer(test_file)
            writer.writerow(data)
        else:
            writer = csv.writer(train_file)
            writer.writerow(data)
