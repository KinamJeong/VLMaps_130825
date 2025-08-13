import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.color import label2rgb
import pandas as pd
from PIL import Image
import os

# 파일 경로 설정 (로컬 환경 기준)
input_csv_path = "door_detected/semantic_map_with_doors_13.csv"
result_img_path = "room_detected/valid_rooms_overlay.png"
room_stats_csv_path = "room_detectedvalid_room_stats.csv"

rmin=270
rmax=523
cmin=350
cmax=517
# Load the map with doors
semantic_map_with_doors = np.loadtxt(input_csv_path, delimiter=",", dtype=int)

# Define constants
EMPTY = -1
DOOR = 16
MIN_ROOM_SIZE = 400

# Create binary map of empty space (excluding doors)
empty_space = (semantic_map_with_doors == EMPTY).astype(int)

# Label connected regions
labeled_map, num_features = label(empty_space)

# Count pixels in each label
unique, counts = np.unique(labeled_map, return_counts=True)
room_areas = dict(zip(unique, counts))

# Filter out small rooms
valid_labels = [label for label, count in room_areas.items() if count > MIN_ROOM_SIZE and label != 0]

# Create visualization
filtered_map = np.isin(labeled_map, valid_labels).astype(int) * labeled_map
room_overlay = label2rgb(filtered_map, bg_label=0)
cropped_overlay = room_overlay[rmin:rmax+1,cmin:cmax+1]

# Save result image
Image.fromarray((cropped_overlay * 255).astype(np.uint8)).save("door_detected/cropped_valid_rooms_overlay.png")

# Save room statistics to CSV
room_stats = [{"Room Label": label, "Area (pixels)": room_areas[label]} for label in valid_labels]
room_stats_df = pd.DataFrame(room_stats)
room_stats_df.to_csv(room_stats_csv_path, index=False)

semantic_output_path = "semantic_topdown_map.csv"
np.savetxt(semantic_output_path, semantic_map_with_doors, fmt="%d", delimiter=",")

# Save filtered room label map (유효한 방만 포함, 나머지는 0)
filtered_map_output_path = "valid_room_index_map.csv"
np.savetxt(filtered_map_output_path, filtered_map, fmt="%d", delimiter=",")

# 출력 경로 확인
print(f"[INFO] 결과 이미지 저장 위치: {os.path.abspath(result_img_path)}")
print(f"[INFO] 방 통계 CSV 저장 위치: {os.path.abspath(room_stats_csv_path)}")
print(f"[INFO] semantic_topdown_map 저장 위치: {os.path.abspath(semantic_output_path)}")
print(f"[INFO] valid_room_index_map 저장 위치: {os.path.abspath(filtered_map_output_path)}")
