# import numpy as np

# # 이미지 크기
# image_width = 512
# image_height = 512

# # y 변수 설정
# y = 6  # y 값을 변경하여 원하는 등분 수를 얻을 수 있습니다.

# # 중심 좌표 계산
# center_x = image_width // 2
# center_y = image_height // 2

# # y에 따른 샘플링 좌표 계산
# if y == 1:
#     sampling_coords = np.array([[center_x, center_y]])
# else:
#     sub_divisions = y + 2
#     x_coords = np.linspace(0, image_width, sub_divisions)
#     y_coords = np.linspace(0, image_height, sub_divisions)

#     x_mesh, y_mesh = np.meshgrid(x_coords[1:-1], y_coords[1:-1])
#     sampling_coords = np.column_stack((x_mesh.ravel(), y_mesh.ravel())).astype(int)

# # 결과 출력
# print(sampling_coords)

# ####################################################################################################################################################################
# import numpy as np
# import matplotlib.pyplot as plt

# # 이미지 크기
# image_width = 512
# image_height = 512

# # y 변수 설정
# y = 32  # y 값을 변경하여 원하는 등분 수를 얻을 수 있습니다.

# # 중심 좌표 계산
# center_x = image_width // 2
# center_y = image_height // 2

# # y에 따른 샘플링 좌표 계산
# if y == 1:
#     sampling_coords = np.array([[center_x, center_y]])
# else:
#     sub_divisions = y + 2
#     x_coords = np.linspace(0, image_width, sub_divisions)
#     y_coords = np.linspace(0, image_height, sub_divisions)

#     x_mesh, y_mesh = np.meshgrid(x_coords[1:-1], y_coords[1:-1])
#     sampling_coords = np.column_stack((x_mesh.ravel(), y_mesh.ravel())).astype(int)

# # 시각화
# plt.scatter(sampling_coords[:, 0], sampling_coords[:, 1], c='r', marker='o', s=0.1)
# plt.xlim(0, image_width)
# plt.ylim(0, image_height)
# plt.gca().invert_yaxis()  # 좌측 상단을 원점으로 설정
# # plt.grid(True)
# plt.savefig(f"sampling_coords_{y}.png", dpi=300, bbox_inches='tight')
# ####################################################################################################################################################################

# import numpy as np


# def cal_results(matrix):
#     epsilon = 1e-15
#     shape = np.shape(matrix)
#     number = 0
#     sum = 0
#     # AA = np.zeros([shape[0]], dtype=np.float)
#     for i in range(shape[0]):
#         number += matrix[i, i]
#         # try:
#         #     # AA[i] = matrix[i, i] / (np.sum(matrix[i, :]) + epsilon)
#         #     AA[i] = matrix[i, i] / np.sum(matrix[i, :])
#         # except ZeroDivisionError:
#         #     AA[i] = 0
#         #     print("!!!!!!!!!!!!!!!!!!!!zero division!!!!!!!!!!!!!!!!!!!!")
#         sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
#     OA = number / np.sum(matrix)
#     # AA_mean = np.mean(AA)
#     pe = sum / (np.sum(matrix) ** 2)
#     Kappa = (OA - pe) / (1 - pe)
#     # return OA, AA_mean, Kappa, AA
#     return OA, Kappa


# # def output_metric(tar, pre):
# #     matrix = confusion_matrix(tar, pre, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
# #     # OA, AA_mean, Kappa, AA = cal_results(matrix)
# #     labels=["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0",]
# #     plot_confusion_matrix(matrix, labels = labels)
# #     # plt.savefig("cm.jpg")
# #     OA, Kappa = cal_results(matrix)
# #     # return OA, AA_mean, Kappa, AA
# #     return OA, Kappa #, matrix
# # #-------------------------------------------------------------------------------
# matrix=np.array([[0,0,0],[0,30,2],[0,0,3]])
# OA, Kappa = cal_results(matrix)
# print(Kappa)


# import numpy as np

# # (256, 256, 3) 크기의 흑백 이미지(값이 동일) 생성 예시
# img = np.zeros((256, 256, 3), dtype=np.uint8)  # 흑백 이미지 생성
# img[10:20, 10:20, :] = 30  # 모든 채널에 값 30 적용
# # 특정 위치에 값 30 적용 (예시에서는 50x50 위치)
# img[50, 50, :] = 30

# # 노란색(RGB: [255, 255, 0])으로 값이 30인 부분을 변경
# yellow_indices = np.all(img == [30, 30, 30], axis=2)  # 값이 30인 부분 찾기
# img[yellow_indices] = [255, 255, 0]  # 노란색으로 변경

# img

# 결과 확인
# 여기서는 이미지를 시각화하여 눈으로 확인하는 대신에, 이미지가 생성되었다고 가정합니다.

import os
import numpy as np
import utils.utils as util
os.chdir('/root/work/hjr/nia-hjr')
image_folder = r'./output/231123_GCP_RE_cosinealr_p3_bp3_clip4000_mirror_sp64_lr001_adam_tb256/result_npy'

cnt = 0
for root, dirs, files in os.walk(image_folder):
    # if "02.수중 및 지상 초분광" in root:
        for image_filename in files:
            if image_filename.endswith('.npy'):
                cnt +=1

result = []
cnt = 0
for root, dirs, files in os.walk(image_folder):
    # if "02.수중 및 지상 초분광" in root:
        for image_filename in files:
            if image_filename.endswith('.npy'):
                temp = np.load(os.path.join(image_folder, image_filename))
                result.append(temp)
                print(cnt)
                cnt +=1

temp = np.array(result)
temp = temp.transpose(1, 2, 0).reshape(2, -1)
OA, kappa = util.output_metric_with_savefig(temp[0,:], temp[1,:], 'output/231123_GCP_RE_cosinealr_p3_bp3_clip4000_mirror_sp64_lr001_adam_tb256')
print(f'Total : OA:{OA:.5f} Kappa:{kappa:.5f}')