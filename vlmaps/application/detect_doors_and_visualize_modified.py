import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from pathlib import Path
from PIL import Image

# ---------------------------
# 파일 경로 설정
input_csv_path = "2d_semantic_topdown/semantic_map_17d_2d_high_h0.csv"     # 입력 CSV
output_img_path = "door_detected/door_detected_map_17d.png"           # 시각화 이미지
output_csv_path = "door_detected/semantic_map_with_doors_17d.csv" 
# ---------------------------

# Load the semantic map
semantic_map = np.loadtxt(input_csv_path, delimiter=",", dtype=int)

# Constants
rmin, rmax = 377, 542 #270,523  / 377, 542
cmin, cmax = 371, 695 #350,517 / 371, 695
max_doors = 20
door_index = 16
empty_index = -1

# Slice the map
cropped_map = semantic_map[rmin:rmax+1, cmin:cmax+1].copy()
H, W = cropped_map.shape

# Distance transform for fast nearest search
binary_map = (cropped_map != empty_index).astype(np.uint8)
distance = distance_transform_edt(1 - binary_map)

# Candidate empty points with sufficient distance
candidates = np.argwhere((cropped_map == empty_index) & (distance > 5))

door_count = 0
marked = np.zeros_like(cropped_map, dtype=bool)
doors = []  # 문 정보 저장 리스트: {'center': (mx, my), 'line': [(x,y), ...]}

log_file = open("detect_log.txt", "a")
for mx, my in candidates:
    if door_count >= max_doors:
        break
    if marked[mx, my]:
        continue

    # Find nearest non-empty pixel A
    ax, ay = -1, -1
    max_radius = 20
    found = False
    log_file.write(f"[DEBUG] 중심점 M: ({mx}, {my})\n")
    for radius in range(1, max_radius + 1):
        log_file.write(f"[DEBUG] 반지름 탐색: radius = {radius}\n")
        for dx in range(-radius, radius + 1):
            dy = radius - abs(dx)
            for sign in [-1, 1]:
                x, y = mx + dx, my + sign * dy
                if 0 <= x < H and 0 <= y < W:
                    if cropped_map[x, y] == 7:
                        ax, ay = x, y
                        found = True
                        log_file.write(f"[DEBUG] 최근접 7번 픽셀 A 발견 → A: ({ax}, {ay})\n")
                        break
            if found:
                break
        if found:
            break
        
    min_dist = np.inf 
    ax, ay = -1, -1
    for dx in range(-20, 21):
        for dy in range(-20, 21):
            x, y = mx + dx, my + dy
            if 0 <= x < H and 0 <= y < W and cropped_map[x, y] != empty_index:
                dist = (x - mx)**2 + (y - my)**2
                if dist < min_dist:
                    min_dist = dist
                    ax, ay = x, y

    if ax == -1:
        log_file.write("[DEBUG] 유효한 7번 픽셀 A를 찾지 못했음 → continue\n")
        continue
        continue

    # 반대 방향 B 계산
    dx, dy = mx - ax, my - ay
    bx, by = mx + dx, my + dy
    log_file.write(f"[DEBUG] 반대 방향 벡터: dx={dx}, dy={dy}\n")
    log_file.write(f"[DEBUG] 반대 방향 B 후보: ({bx}, {by})\n")
    bx, by = mx + dx, my + dy
    if not (0 <= bx < H and 0 <= by < W):
        log_file.write("[DEBUG] B가 범위를 벗어남 → continue\n")
        continue
        continue
    if cropped_map[bx, by] == empty_index:
        continue

    # 거리 조건 확인
    ab_dist = np.sqrt((ax - bx)**2 + (ay - by)**2)
    ab_dist = np.sqrt((ax - bx)**2 + (ay - by)**2)
    log_file.write(f"[DEBUG] 유효한 문 후보 발견: A=({ax},{ay}), M=({mx},{my}), B=({bx},{by}), B 인덱스={cropped_map[bx, by]}\n")
    if 10 <= ab_dist <= 40:
        # 선분 마킹 및 저장
        line_pixels = []
        num_pts = int(np.ceil(ab_dist))
        for i in range(num_pts + 1):
            px = int(round(ax + (bx - ax) * i / num_pts))
            py = int(round(ay + (by - ay) * i / num_pts))
            if 0 <= px < H and 0 <= py < W:
                cropped_map[px, py] = door_index
                marked[px, py] = True
                line_pixels.append((px, py))

        doors.append({'center': (mx, my), 'line': line_pixels})
        door_count += 1
    log_file.write(f"[DEBUG] 문 개수 증가: 현재 {door_count}\n")

# 문 중심점 간 거리 기반 중복 제거
to_remove = set()
for i in range(len(doors)):
    for j in range(i + 1, len(doors)):
        cx1, cy1 = doors[i]['center']
        cx2, cy2 = doors[j]['center']
        dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        if dist <= 13:
            to_remove.add(j)  # 나중에 추가된 문 제거

# 제거 수행
for idx in sorted(to_remove, reverse=True):
    for (x, y) in doors[idx]['line']:
        cropped_map[x, y] = empty_index
        marked[x, y] = False
    doors.pop(idx)

# 전체 semantic_map에 문 정보 반영
semantic_map[rmin:rmax+1, cmin:cmax+1] = cropped_map

# 시각화용 RGB 이미지 생성
rgb_image = np.zeros((H, W, 3), dtype=np.uint8)
rgb_image[cropped_map == empty_index] = [0, 0, 0]       # 검정
rgb_image[cropped_map == door_index] = [0, 0, 255]       # 파랑
rgb_image[(cropped_map != empty_index) & (cropped_map != door_index)] = [160, 160, 160]  # 회색

# 문 중심점 시각화 (빨간 점)
for door in doors:
    x, y = door['center']
    if 0 <= x < H and 0 <= y < W:
        rgb_image[x, y] = [255, 0, 0]

# 이미지 저장
Image.fromarray(rgb_image).save(output_img_path)
print(f"[INFO] 이미지 저장 완료: {output_img_path}")

# 문 정보 포함 CSV 저장
np.savetxt(output_csv_path, semantic_map, fmt="%d", delimiter=",")
log_file.close()
print(f"[INFO] 문 정보 포함된 CSV 저장 완료: {output_csv_path}")