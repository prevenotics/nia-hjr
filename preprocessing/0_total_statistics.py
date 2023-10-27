import matplotlib.pyplot as plt
import os
import cv2
import tifffile
import numpy as np

#전체 tif 파일의 히스토그램 통계

total_hist = []
# TIFF 이미지 파일을 열고 numpy 배열로 변환하는 함수
def open_tiff_as_numpy(file_path):
    image = tifffile.imread(file_path)
    return image

def plot_histogram(image_path):
    # image = cv2.imread(image_path, 0)
    image = open_tiff_as_numpy(image_path)
    
    # hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # hist = cv2.calcHist([image], [0], None, [65536], [0, 65536])
    hist,bins = np.histogram(image, bins = 65536, range=(0,65536))
    return hist
    # plt.plot(hist)
    # plt.title('Histogram')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.show()
i=0
def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:            
            if file.endswith('.tif') and "RE" in file:
                global hist_sum, i
                image_path = os.path.join(root, file)
                hist = plot_histogram(image_path)
                hist_sum += hist.astype(np.uint64).flatten()
                i+=1
                print(i)
                
                
                
                
                
                

hist_sum = np.zeros((65536,), dtype=np.uint64)
folder_path = r'D:/DATA/hjr/dataset/1.원천데이터'
process_folder(folder_path)
np.save('hist_RE', hist_sum)
plt.plot(hist_sum)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
plt.savefig('histogram.png')