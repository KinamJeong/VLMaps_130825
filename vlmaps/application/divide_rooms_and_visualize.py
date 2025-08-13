import cv2
import numpy as np
from collections import defaultdict, Counter
# 시맨틱 마스크 팔레트 (RGB → 객체 이름)
palette = {
    (255, 0, 0): "table",
    (0, 255, 0): "sofa",
    (0, 0, 255): "bed",
    (255, 255, 0): "sink",
    (255, 0, 255): "chair",
    (0, 255, 255): "toilet",
    (255, 128, 0): "tv_monitor",
    (128, 0, 255): "wall",
    (135, 206, 235): "window",
    (192, 192, 192): "cabinet"
}
# 객체 → 후보 방들
object_to_room_votes = {
    "bed": ["bedroom"],
    "sofa": ["living room"],
    "table": ["dining room", "living room"],
    "toilet": ["bathroom"],
    "sink": ["bathroom", "kitchen"],
    "tv_monitor": ["living room", "bedroom"],
    "chair": ["dining room", "living room"],
    "window": ["living room", "shared area"],
    "cabinet": ["kitchen"],
    "wall": ["corridor"]
}
# 특수 색상 정의
WALL_COLOR = (255, 0, 0)
IGNORE_COLOR = (0, 0, 0)
# 이미지 불러오기
semantic = cv2.imread("semantic_mask.png")
room_mask = cv2.imread("door_detected/cropped_valid_rooms_overlay.png")
room_mask = cv2.resize(room_mask, (semantic.shape[1], semantic.shape[0]), interpolation=cv2.INTER_NEAREST)
room_mask_rgb = cv2.cvtColor(room_mask, cv2.COLOR_BGR2RGB)
h, w = semantic.shape[:2]
room_ids = np.unique(room_mask_rgb.reshape(-1, 3), axis=0)
# 방별 객체 수집
room_obj_counter = defaultdict(Counter)
for y in range(h):
    for x in range(w):
        obj_color = tuple(semantic[y, x][::-1])  # BGR → RGB
        room_color = tuple(room_mask_rgb[y, x])
        obj_name = palette.get(obj_color)
        if obj_name and room_color != WALL_COLOR and room_color != IGNORE_COLOR:
            room_obj_counter[room_color][obj_name] += 1
# 방 이름 추론
room_name_map = {}
for room_color, obj_count in room_obj_counter.items():
    room_vote_counter = Counter()
    for obj_name, count in obj_count.items():
        candidate_rooms = object_to_room_votes.get(obj_name, [])
        for room in candidate_rooms:
            room_vote_counter[room] += count
    final_room = room_vote_counter.most_common(1)[0][0] if room_vote_counter else "Unknown"
    room_name_map[room_color] = final_room
    print(f"Room {room_color} → {final_room} (votes: {room_vote_counter})")
# 시각화
output = room_mask_rgb.copy()
for room_color in room_ids:
    if tuple(room_color) == IGNORE_COLOR:
        continue  # 검정색 영역은 라벨 생략
    mask = np.all(room_mask_rgb == room_color, axis=-1).astype(np.uint8)
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if tuple(room_color) == WALL_COLOR:
            room_name = "wall"
        else:
            room_name = room_name_map.get(tuple(room_color), "Unknown")
        cv2.putText(output, room_name, (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(output, room_name, (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite("voted_room_labels.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))