import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist

# --- 파일 경로 및 설정 ---
semantic_map_path = "semantic_map_RP_2d_majority_tv_override_0717.csv"
output_csv_path = "room_assignment_RP_all_objects_0805.csv"

# --- Matterport3D trimmed category list ---
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
    "carpet",
    "bookcase",
    "desktop_computer",
    "dishrag",
    "carpet",
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
mp3dcat_trimmed = mp3dcat[1:-1]

# --- Room type ID mapping ---
room_name_to_id = {
    "bathroom": 1,
    "bedroom": 2,
    "corridor": 3,
    "living room": 4,
    "study room": 5,
    "kitchen": 6,
}

# --- Object to room vote mapping ---
object_to_room_votes = {
    "table": ["living room", "kitchen", "study room"],
    "sofa": ["living room"],
    "bed": ["bedroom"],
    "sink": ["bathroom", "kitchen"],
    "chair": ["living room", "study room", "bedroom"],
    "toilet": ["bathroom"],
    "tv_monitor": ["living room", "bedroom"],
    "window": ["bathroom", "bedroom", "living room", "kitchen", "study room"],
    "shelving": ["bathroom", "bedroom", "living room", "kitchen", "study room"],
    "cabinet": ["bathroom", "kitchen", "bedroom"],
    "curtain": ["bathroom", "bedroom", "living room", "study room"],
    "desktop_computer": ["study room"],
    "counter": ["kitchen", "bathroom"],
    "stove": ["kitchen"],
    "book": ["study room", "bedroom"],
    "refrigerator": ["kitchen"],
    "microwave": ["kitchen"],
    "towel": ["bathroom"],
    "mirror": ["bathroom", "bedroom"]
}

# --- Area thresholds for specific objects ---
area_threshold_map = {
    "refrigerator": 30,
    "desktop_computer": 4,
    "counter": 10,
    "stove": 20,
    "toilet": 10,
    "microwave":10,
    "cabinet":10,
    "shelving":10,
    "mirror":10,
    "towel":10,
    "book":10,
    "bed":100,
}
default_threshold = 50

semantic_map = np.loadtxt(semantic_map_path, delimiter=",", dtype=int)
H, W = semantic_map.shape
room_map = np.zeros((H, W), dtype=int)

# 단일 성격 객체 할당
single_object_centers = {}
for obj_name, votes in object_to_room_votes.items():
    if len(votes) != 1:
        continue
    if obj_name not in mp3dcat_trimmed:
        continue
    obj_id = mp3dcat_trimmed.index(obj_name)
    room_id = room_name_to_id[votes[0]]
    room_map[semantic_map == obj_id] = room_id

    # 중심점 저장 및 출력
    threshold = area_threshold_map.get(obj_name, default_threshold)
    mask = (semantic_map == obj_id).astype(int)
    labeled, num = label(mask)
    centers = []
    count = 0
    for i in range(1, num + 1):
        region_mask = (labeled == i)
        if np.sum(region_mask) < threshold:
            continue
        center = center_of_mass(region_mask)
        center_int = tuple(map(int, center))
        centers.append(center_int)
        count += 1
    print(f"[INFO] '{obj_name}' 중심점 (면적 ≥ {threshold}) - 총 {count}개:")
    for j, c in enumerate(centers):
        print(f"  #{j+1}: (row={c[0]}, col={c[1]})")
    single_object_centers[obj_name] = centers
    print()

# 다중 성격 객체 중심점 분류 함수
multi_obj_centers_by_room = {}
def assign_room_to_multi_vote_objects(room_map_base, object_name, area_threshold=50):
    obj_id = mp3dcat_trimmed.index(object_name)
    obj_mask = (semantic_map == obj_id).astype(int)
    labeled_obj, num_obj = label(obj_mask)

    obj_centers = []
    obj_instance_masks = []

    print(f"\n 카테고리 : '{object_name}'")
    for i in range(1, num_obj + 1):
        instance_mask = (labeled_obj == i)
        if np.sum(instance_mask) < area_threshold:
            continue
        center = center_of_mass(instance_mask)
        center_int = tuple(map(int, center))
        obj_centers.append(tuple(map(int, center)))
        obj_instance_masks.append(instance_mask)
        print(f" - 개별 객체 {i} : 중심점 (row={center_int[0]}, col={center_int[1]})")

    all_reference_centers = []
    ref_center_info = [] 
    for obj_name_ref, centers in single_object_centers.items():
        if obj_name_ref not in object_to_room_votes:
            continue
        votes = object_to_room_votes[obj_name_ref]
        if len(votes) != 1:
            continue
        room_id = room_name_to_id[votes[0]]
        for c in centers:
            all_reference_centers.append((c, room_id))
            ref_center_info.append((c, obj_name_ref, room_id))

    for obj_name_ref, centers in multi_obj_centers_by_room.items():
        for c, room_id in centers:
            all_reference_centers.append((c, room_id))
            ref_center_info.append((c, obj_name_ref, room_id))

    updated_map = room_map_base.copy()
    new_centers_with_rooms = []
    for idx, center in enumerate(obj_centers):
        point = np.array([center])
        sorted_refs = sorted(all_reference_centers, key=lambda x: np.linalg.norm(point - np.array(x[0])))

        ref_point, ref_room_id = sorted_refs[0]
        updated_map[obj_instance_masks[idx]] = ref_room_id
        new_centers_with_rooms.append((center, ref_room_id))

        for c, ref_obj_name, room_id in ref_center_info:
            if c == ref_point:
                room_name = [k for k, v in room_name_to_id.items() if v == room_id][0]
                print(f"  → 중심점 (row={center[0]}, col={center[1]})는 \n'{ref_obj_name}'의 중심점 (row={ref_point[0]}, col={ref_point[1]})에 가장 가까움")
                print(f"     할당된 방: {room_name} (room_id={room_id})\n")
                break
    return updated_map, new_centers_with_rooms

# 분류 대상 객체들
multi_vote_objects = [k for k, v in object_to_room_votes.items() if len(v) > 1]
room_map_final = room_map.copy()
for obj_name in multi_vote_objects:
    room_map_final, new_centers = assign_room_to_multi_vote_objects(room_map_final, obj_name, area_threshold=50)
    multi_obj_centers_by_room[obj_name] = new_centers

# 저장
np.savetxt(output_csv_path, room_map_final, fmt="%d", delimiter=",")
print(f"[INFO] Saved room assignment map to: {output_csv_path}")
