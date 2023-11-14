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

#####################################################################################################################################################################
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
#####################################################################################################################################################################

import numpy as np


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


# def output_metric(tar, pre):
#     matrix = confusion_matrix(tar, pre, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
#     # OA, AA_mean, Kappa, AA = cal_results(matrix)
#     labels=["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0",]
#     plot_confusion_matrix(matrix, labels = labels)
#     # plt.savefig("cm.jpg")
#     OA, Kappa = cal_results(matrix)
#     # return OA, AA_mean, Kappa, AA
#     return OA, Kappa #, matrix
# #-------------------------------------------------------------------------------
matrix=np.array([[0,0,0],[0,30,2],[0,0,3]])
OA, Kappa = cal_results(matrix)
print(Kappa)
