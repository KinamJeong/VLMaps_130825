import numpy as np
import matplotlib.pyplot as plt

# npy 파일 경로
file_path = 'grid_1_clip.npy'

# npy 파일 로드
data = np.load(file_path)

# 데이터 정보 출력
print("Data shape:", data.shape)
print("Data type:", data.dtype)

# 시각화
plt.imshow(data, cmap='viridis')  # 또는 'gray', 'plasma', 'jet' 등 원하는 colormap
plt.colorbar()
plt.title('Visualization of NPY Data')
plt.show()
