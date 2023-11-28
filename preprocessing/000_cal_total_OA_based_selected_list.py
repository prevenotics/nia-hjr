import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

os.chdir('/root/work/hjr/nia-hjr/preprocessing')

def output_metric_with_savefig(tar, pre, path, ignore_class=30):
    # matrix = confusion_matrix(tar, pre, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30])
    # matrix = matrix[:30,:30] #ignore_class = 30
    # OA, Kappa = cal_results(matrix)
    tar_no_ignore = tar[tar !=ignore_class]
    pre_no_ignore = pre[tar !=ignore_class]
    matrix = confusion_matrix(tar_no_ignore, pre_no_ignore, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
    # OA, AA_mean, Kappa, AA = cal_results(matrix)
    
    ####################################################################################################################################################################
    labels=["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"]
    plot_confusion_matrix2(matrix, labels, path)
    ###################################################################################################################################################################
    
    OA, Kappa = cal_results(matrix)
    # return OA, AA_mean, Kappa, AA
    return OA, Kappa #, matrix
#-------------------------------------------------------------------------------

def cal_results(matrix):
    epsilon = 1e-15
    shape = np.shape(matrix)
    number = 0
    sum = 0
    # AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]): 
        number += matrix[i, i]
        # try:
        #     # AA[i] = matrix[i, i] / (np.sum(matrix[i, :]) + epsilon)
        #     AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        # except ZeroDivisionError:
        #     AA[i] = 0
        #     print("!!!!!!!!!!!!!!!!!!!!zero division!!!!!!!!!!!!!!!!!!!!")
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    # AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    # return OA, AA_mean, Kappa, AA
    return OA, Kappa
def plot_confusion_matrix(con_mat, labels, path, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.figure(figsize=[25,25])
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(path, 'cm.png'))
    plt.clf()

def plot_confusion_matrix2(con_mat, labels, path, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=True):
    plt.figure(figsize=[25, 25])

    if normalize:
        con_mat = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 클래스별로 정규화

    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    marks = np.arange(len(labels))
    nlabels = []

    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}(n={1})'.format(labels[k], n)
        nlabels.append(nlabel)

    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 2.

    for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
        if normalize:
            plt.text(j, i, '{0:.2f}'.format(con_mat[i, j]), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        else:
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(path, 'cm.png'))
    plt.clf()


# LU_RE_selected.txt 파일명과 npy 파일들이 위치한 디렉토리
selected_file = 'LU_RE_selected.txt'
npy_directory = '/root/work/hjr/nia-hjr/output/231123_GCP_RE_cosinealr_p3_bp3_clip4000_mirror_sp64_lr001_adam_tb256/result_npy/'
prefix='2023-11-26-09-54-47_'
# 선택된 파일 경로를 읽어와서 해당 npy 파일들을 로드하여 리스트에 추가
npy_data = []

with open(selected_file, 'r') as file:
    for line in file:
        path = line.strip()  # 개행 문자 제거
        file_name = os.path.basename(path)  # 파일 이름 추출
        
        npy_file_path = os.path.join(npy_directory, prefix+file_name.replace('.mat', '.npy'))
        
        if os.path.exists(npy_file_path):
            npy_content = np.load(npy_file_path)
            npy_data.append(npy_content)

# 모든 데이터를 NumPy 배열로 변환
final_array = np.array(npy_data)
final_array = final_array.transpose(1, 2, 0).reshape(2, -1)
OA, kappa = output_metric_with_savefig(final_array[0,:], final_array[1,:], './')
print(f'========Total : OA:{OA:.5f} Kappa:{kappa:.5f}========')
