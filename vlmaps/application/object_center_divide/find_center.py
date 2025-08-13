import numpy as np
from scipy.ndimage import label, center_of_mass

# ---------- 설정 ----------
semantic_map_path = "semantic_map_RP_2d_majority_tv_override_0717.csv"

# 단일 성격 객체 리스트
single_room_objects = [
    "bed", "sofa", "toilet", "window", "cabinet", "stove",
    "refrigerator", "shelving", "curtain",
    "desktop_computer", "counter", "microwave"
]

# 객체별 면적 임계값 설정
area_threshold_map = {
    "refrigerator": 30,
    "desktop_computer": 3,
    "counter": 10,
    "stove": 20,
    "toilet": 10,
    "microwave" : 10
}

default_threshold = 50  # 그 외 객체의 기본 면적 임계값
# --------------------------

# mp3dcat 정의 및 인덱스 기준 수정
mp3dcat = [
    "void", "table", "sofa", "bed", "sink", "chair", "toilet", "tv_monitor", "wall", "window", "clothes", "picture",
    "cushion", "coffeepot", "appliances", "tree", "shelving", "door", "cabinet", "curtain", "chest_of_drawers",
    "printer", "bookcase", "desktop_computer", "dishrag", "column", "study_desk", "counter", "oscilloscope",
    "lighting", "stove", "book", "blinds", "refrigerator", "seating", "microwave", "furniture", "towel", "mirror", "shower"
]
mp3dcat_trimmed = mp3dcat[1:-1]

# semantic map 불러오기
semantic_map = np.loadtxt(semantic_map_path, delimiter=",", dtype=int)

# 객체 중심점 추출 및 출력
for obj_name in single_room_objects:
    if obj_name not in mp3dcat_trimmed:
        print(f"[WARN] '{obj_name}' not found in mp3dcat[1:-1], skipping.")
        continue

    obj_id = mp3dcat_trimmed.index(obj_name)
    threshold = area_threshold_map.get(obj_name, default_threshold)

    mask = (semantic_map == obj_id).astype(int)
    labeled, num = label(mask)

    centers = []
    for i in range(1, num + 1):
        region_mask = (labeled == i)
        if np.sum(region_mask) < threshold:
            continue
        center = center_of_mass(region_mask)
        centers.append(tuple(map(int, center)))

    print(f"\n[INFO] '{obj_name}' 중심점 (면적 ≥ {threshold}) - 총 {len(centers)}개:")
    for j, c in enumerate(centers):
        print(f"  #{j+1}: (row={c[0]}, col={c[1]})")
