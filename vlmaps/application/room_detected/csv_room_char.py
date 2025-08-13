import numpy as np
import pandas as pd
import cv2
import csv
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib import colormaps

def analyze_room_candidates(room_obj_counter, room_mask, semantic, mp3dcat, object_to_room_votes, room_area_threshold=10000, vote_threshold=0.25):
    from collections import Counter
    import numpy as np

    h, w = semantic.shape
    room_center_map = {} 
    print("\n=== 복합 후보 방 분석 시작 ===")

    for room_idx, obj_count in room_obj_counter.items():
        # 1. 방 크기 기준 필터
        pixel_count = np.sum(room_mask == room_idx)
        if pixel_count < room_area_threshold:
            continue

        # 2. 투표 집계
        vote = Counter()
        for obj_name, count in obj_count.items():
            for room_type in object_to_room_votes.get(obj_name, []):
                weight = room_vote_weights.get(obj_name, 1)
                vote[room_type] += count*weight
        total_votes = sum(vote.values())
        top_votes = [(room, v) for room, v in vote.items() if v / total_votes >= vote_threshold]

        if len(top_votes) < 2 or "corridor" in [r for r, _ in top_votes]:
            continue

        print(f"\n[Room {room_idx}] 복합 공간 후보 (크기={pixel_count} pixels)")
        print("Top 후보 방들:")
        for r, v in top_votes:
            print(f"  - {r}: {v} votes ({v/total_votes:.1%})")

        # 3. 각 방에만 있는 객체 추출
        candidates = [r for r, _ in top_votes]
        candidate_to_obj = {r: set() for r in candidates}
        for obj, rooms in object_to_room_votes.items():
            for r in candidates:
                if r in rooms:
                    candidate_to_obj[r].add(obj)

        unique_objs = {}
        for r in candidates:
            other_objs = set().union(*(candidate_to_obj[o] for o in candidates if o != r))
            unique_objs[r] = candidate_to_obj[r] - other_objs
        room_obj_centers = {}
        # 4. 객체 중심 좌표 계산
        for r in candidates:
            print(f"  >> {r} 전용 객체: {sorted(unique_objs[r])}")
            coords = []
            for y in range(h):
                for x in range(w):
                    if room_mask[y, x] == room_idx:
                        obj_idx = semantic[y, x] + 1
                        if 0 < obj_idx < len(mp3dcat):
                            obj_name = mp3dcat[obj_idx]
                            if obj_name in unique_objs[r]:
                                coords.append((y, x))
            if coords:
                ys, xs = zip(*coords)
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                room_obj_centers[r]=(cy,cx)
                print(f"    → {r} 전용 객체 중심 좌표: ({cy}, {cx})")
            else:
                print(f"    → {r} 전용 객체 중심 좌표 없음")
        room_center_map[room_idx] = room_obj_centers
        # 5. 방 전체 중심 좌표
        yx_coords = np.argwhere(room_mask == room_idx)
        cy_all, cx_all = np.mean(yx_coords, axis=0).astype(int)
        print(f"  >> Room {room_idx} 전체 중심 좌표: ({cy_all}, {cx_all})")

    return room_center_map

def split_room_by_centers(label_index_map, room_mask, room_idx, center_map, target_labels):
    coords = np.argwhere(room_mask == room_idx)
    for (y, x) in coords:
        pos = np.array([y, x])
        dists = {k: np.linalg.norm(pos - np.array(v)) for k, v in center_map.items()}
        closest = min(dists, key=dists.get)
        label_index_map[y, x] = target_labels[closest]       

# Matterport3D 카테고리 (전체 41개)
mp3dcat = [
    "void",
    "table",
    "sofa",
    "bed",
    "sink",
    "chair",
    "toilet",
    "tv_monitor",
    "wall",
    "window",
    "clothes",
    "picture",
    "cushion",
    "coffeepot",
    "appliances",
    "tree",
    "shelving",
    "door",
    "cabinet",
    "curtain",
    "chest_of_drawers",
    "printer",
    "bookcase",
    "desktop_computer",
    "dishrag",
    "column",
    "study_desk",
    "counter",
    "oscilloscope",
    "lighting",
    "stove",
    "book",
    "blinds",
    "refrigerator",
    "seating",
    "microwave",
    "furniture",
    "towel",
    "mirror",
    "shower",
]

# 객체 → 후보 방들
object_to_room_votes = {
    "bed": ["bedroom"],
    "sofa": ["living room"],
    "table": ["living room", "study room"],
    "toilet": ["bathroom"],
    "sink": ["bathroom", "kitchen"],
    "tv_monitor": ["living room", "bedroom","study room"],
    "chair": ["living room", "study room"],
    "window": ["living room"],
    "cabinet": ["study room"],
    "wall": ["corridor"],
    "stove": ["kitchen"],
    "refrigerator": ["kitchen"],
    "shower": ["bathroom"],
    "shelving": ["study room"],
    "curtain": ["bedroom"],
    
    "desktop_computer":["study room"],
    "counter":["kitchen"],
    "microwave":["kitchen"],
}

room_vote_weights = {
    "bed": 3,
    "sofa" : 3,
    "desktop_computer": 30,
    "counter": 20,
    "microwave": 25,
    "stove": 20,
    "refrigerator": 20,
    "toilet": 20
}


# CSV 파일 로딩
semantic = pd.read_csv("semantic_map_RP_2d_majority_tv_override_0717.csv", header=None).to_numpy()
room_mask = pd.read_csv("valid_room_index_map.csv", header=None).to_numpy()
h, w = semantic.shape



# 방별 객체 수집 (인덱스 보정 포함)
room_obj_counter = defaultdict(Counter)
for y in range(h):
    for x in range(w):
        obj_idx = semantic[y, x] + 1  # mp3dcat[1] == table 이므로 보정
        room_idx = room_mask[y, x]

        if 0 < obj_idx < len(mp3dcat) - 1:  # 'void' 제외, 마지막 dummy 제외
            obj_name = mp3dcat[obj_idx]
            if obj_name in object_to_room_votes:
                room_obj_counter[room_idx][obj_name] += 1

# 방 이름 추론
room_name_map = {}
for room_idx, obj_count in room_obj_counter.items():
    vote = Counter()
    for obj_name, count in obj_count.items():
        for room_type in object_to_room_votes.get(obj_name, []):
            weight = room_vote_weights.get(obj_name, 1) 
            vote[room_type] += count *weight
    
    print(f"[Room {room_idx}] Object votes:")
    for room_type, v_count in vote.most_common():
        print(f"  - {room_type}: {v_count} vote(s)")  

    room_name_map[room_idx] = vote.most_common(1)[0][0] if vote else "Unknown"

#중심좌표 가져오기
room_centers=analyze_room_candidates(
    room_obj_counter=room_obj_counter,
    room_mask=room_mask,
    semantic=semantic,
    mp3dcat=mp3dcat,
    object_to_room_votes=object_to_room_votes
)

# 방 이름 → 인덱스 매핑
room_class_names = sorted(set(room_name_map.values()) | {"Unknown"})
room_class_to_idx = {name: idx for idx, name in enumerate(room_class_names)}

label_index_map = np.full((h, w), room_class_to_idx["Unknown"], dtype=np.uint8)

# 방 크기 출력
room_area = {}
for room_idx in np.unique(room_mask):
    if room_idx in room_name_map:
        room_name = room_name_map[room_idx]
        pixel_count = np.sum(room_mask == room_idx)
        room_area[room_idx] = pixel_count
        print(f"[Room {room_idx}] \"{room_name}\": {pixel_count} pixel(s)")

# 출력용 label map 생성
for room_idx in np.unique(room_mask):
    label = room_name_map.get(room_idx, "Unknown")
    label_index_map[room_mask == room_idx] = room_class_to_idx[label]

if 2 in room_centers and set(room_centers[2].keys()) == {"living room", "kitchen"}:
    print("[INFO] Room 2를 living room(4) / kitchen(6)으로 중심 기반 재분할합니다.")
    split_room_by_centers(
        label_index_map,
        room_mask,
        room_idx=2,
        center_map=room_centers[2],
        target_labels={"living room": 4, "kitchen": 6}
    )

#컬러맵 생성
num_classes = len(room_class_to_idx)
cmap = plt.get_cmap('tab20')
idx_to_color = {
    idx: (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
    for i, idx in enumerate(sorted(room_class_to_idx.values()))
}

# 컬러 이미지 생성
color_image = np.zeros((h, w, 3), dtype=np.uint8)
for idx, color in idx_to_color.items():
    color_image[label_index_map == idx] = color

# 저장
cv2.imwrite("room_labels_colormap_0717_3.png", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
np.savetxt("room_labels_0717_3.csv", label_index_map, fmt='%d', delimiter=',')

with open("room_labels_colormap_0717_3.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "Room Name", "RGB Color (R,G,B)"])
    for name, idx in sorted(room_class_to_idx.items(), key=lambda x: x[1]):
        rgb = idx_to_color[idx]
        writer.writerow([idx, name, f"{rgb[0]},{rgb[1]},{rgb[2]}"])

print("room_labels.csv, room_labels_colormap.png, room_labels_colormap.csv 저장 완료")