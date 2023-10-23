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


import numpy as np
import matplotlib.pyplot as plt

# 이미지 크기
image_width = 512
image_height = 512

# y 변수 설정
y = 15  # y 값을 변경하여 원하는 등분 수를 얻을 수 있습니다.

# 중심 좌표 계산
center_x = image_width // 2
center_y = image_height // 2

# y에 따른 샘플링 좌표 계산
if y == 1:
    sampling_coords = np.array([[center_x, center_y]])
else:
    sub_divisions = y + 2
    x_coords = np.linspace(0, image_width, sub_divisions)
    y_coords = np.linspace(0, image_height, sub_divisions)

    x_mesh, y_mesh = np.meshgrid(x_coords[1:-1], y_coords[1:-1])
    sampling_coords = np.column_stack((x_mesh.ravel(), y_mesh.ravel())).astype(int)

# 시각화
plt.scatter(sampling_coords[:, 0], sampling_coords[:, 1], c='r', marker='o', s=0.1)
plt.xlim(0, image_width)
plt.ylim(0, image_height)
plt.gca().invert_yaxis()  # 좌측 상단을 원점으로 설정
plt.grid(True)
plt.savefig("sampling_coords.png", dpi=300, bbox_inches='tight')