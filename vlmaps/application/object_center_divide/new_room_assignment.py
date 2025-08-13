
import numpy as np
from scipy.ndimage import label, center_of_mass
from typing import Dict, List

# -------------------------
# File paths (adjust if needed)
# -------------------------
SEMANTIC_MAP_PATH = "semantic_map_RP_2d_majority_tv_override_0717.csv"
OUTPUT_CSV_PATH   = "room_assignment_RP_all_objects_scene_adaptive.csv"

# -------------------------
# Matterport3D trimmed category list
# (indexing must match how the CSV was produced)
# -------------------------
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

# -------------------------
# Room type ID mapping
# -------------------------
room_name_to_id = {
    "bathroom": 1,
    "bedroom": 2,
    "corridor": 3,
    "living room": 4,
    "study room": 5,
    "kitchen": 6,
}

# -------------------------
# Object to room vote mapping
# -------------------------
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
    "mirror": ["bathroom", "bedroom"],
}

# -------------------------
# Base thresholds (will be adapted per scene)
# -------------------------
area_threshold_map_base = {
    "refrigerator": 30,
    "desktop_computer": 4,
    "counter": 10,
    "stove": 20,
    "toilet": 10,
    "microwave": 10,
    "cabinet": 10,
    "shelving": 10,
    "mirror": 10,
    "towel": 10,
    "book": 10,
    "bed": 100,
}
DEFAULT_THRESHOLD = 50

# -------------------------
# Scene-adaptive threshold computation (CSV-based)
# -------------------------
MIN_PX, MAX_PX = 5, 200  # safety clamp

def compute_scene_adaptive_thresholds_from_csv(
    semantic_map: np.ndarray,
    mp3dcat_trimmed: List[str],
    base_map: Dict[str, int],
    default_threshold: int = 50,
    percentile: int = 10,
    min_instances: int = 3
) -> Dict[str, int]:
    """
    Compute per-category area thresholds using connected-component areas
    from the CSV (semantic_map). Returns a dict {category_name: threshold_px}.
    - base_map: base thresholds to avoid too-low values
    - percentile: use lower percentile (e.g., 10th) to be robust to outliers
    - min_instances: if fewer instances than this, keep base threshold
    """
    thresholds = {}
    for obj_name in mp3dcat_trimmed:
        obj_id = mp3dcat_trimmed.index(obj_name)

        mask = (semantic_map == obj_id).astype(np.uint8)
        if mask.sum() == 0:
            thresholds[obj_name] = base_map.get(obj_name, default_threshold)
            continue

        labeled, num = label(mask)
        areas = []
        for i in range(1, num + 1):
            area = int((labeled == i).sum())
            if area > 0:
                areas.append(area)

        base = base_map.get(obj_name, default_threshold)
        if len(areas) >= min_instances:
            p = int(np.percentile(areas, percentile))
            thr = max(base, p)  # never go below base
        else:
            thr = base

        thr = int(np.clip(thr, MIN_PX, MAX_PX))
        thresholds[obj_name] = thr

    return thresholds

# -------------------------
# Main
# -------------------------
def main():
    # Load CSV semantic map
    semantic_map = np.loadtxt(SEMANTIC_MAP_PATH, delimiter=",", dtype=int)
    H, W = semantic_map.shape

    # Compute scene-adaptive thresholds and merge with base
    scene_thresholds = compute_scene_adaptive_thresholds_from_csv(
        semantic_map,
        mp3dcat_trimmed,
        area_threshold_map_base,
        default_threshold=DEFAULT_THRESHOLD,
        percentile=10,
        min_instances=3,
    )
    area_threshold_map = area_threshold_map_base.copy()
    area_threshold_map.update(scene_thresholds)
    print("[INFO] Scene-adaptive thresholds (sample):",
          {k: area_threshold_map[k] for k in list(area_threshold_map.keys())[:8]})

    # Room map to fill
    room_map = np.zeros((H, W), dtype=int)

    # --- Pass 1: assign rooms for single-vote categories & collect anchors ---
    single_object_centers = {}
    for obj_name, votes in object_to_room_votes.items():
        if len(votes) != 1:
            continue
        if obj_name not in mp3dcat_trimmed:
            continue

        obj_id = mp3dcat_trimmed.index(obj_name)
        room_id = room_name_to_id[votes[0]]

        # Assign room label to all pixels of this category
        room_map[semantic_map == obj_id] = room_id

        # Compute centers for sufficiently large instances (area threshold per scene)
        threshold = area_threshold_map.get(obj_name, DEFAULT_THRESHOLD)
        mask = (semantic_map == obj_id).astype(np.uint8)
        labeled, num = label(mask)

        centers = []
        count = 0
        for i in range(1, num + 1):
            region_mask = (labeled == i)
            if int(region_mask.sum()) < threshold:
                continue
            center = center_of_mass(region_mask)
            center_int = tuple(map(int, center))
            centers.append(center_int)
            count += 1

        print(f"[INFO] '{obj_name}' centers (area ≥ {threshold}) - {count} found")
        for j, c in enumerate(centers):
            print(f"  #{j+1}: (row={c[0]}, col={c[1]})")

        single_object_centers[obj_name] = centers

    # --- Pass 2: assign rooms for multi-vote categories by nearest anchor ---
    multi_obj_centers_by_room = {}

    def assign_room_to_multi_vote_objects(room_map_base: np.ndarray, object_name: str, area_threshold: int) -> np.ndarray:
        obj_id = mp3dcat_trimmed.index(object_name)
        obj_mask = (semantic_map == obj_id).astype(np.uint8)
        labeled_obj, num_obj = label(obj_mask)

        obj_centers = []
        obj_instance_masks = []

        print(f"\n[INFO] Multi-vote category: '{object_name}' (area ≥ {area_threshold})")
        for i in range(1, num_obj + 1):
            instance_mask = (labeled_obj == i)
            if int(instance_mask.sum()) < area_threshold:
                continue
            center = center_of_mass(instance_mask)
            center_int = tuple(map(int, center))
            obj_centers.append(center_int)
            obj_instance_masks.append(instance_mask)
            print(f"  - instance {i}: center (row={center_int[0]}, col={center_int[1]})")

        # Build reference anchor list: single-vote anchors + previously assigned multi-vote anchors
        all_reference_centers = []  # list of (center_xy, room_id)
        ref_center_info = []        # list of (center_xy, ref_obj_name, room_id)

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

        # If no anchors are available, skip gracefully
        if not all_reference_centers:
            print("[WARN] No reference anchors yet; skipping assignment for this category.")
            return updated_map, new_centers_with_rooms

        for idx, center in enumerate(obj_centers):
            # nearest anchor
            point = np.array([center])
            dists = [np.linalg.norm(point - np.array(x[0])) for x in all_reference_centers]
            nearest_idx = int(np.argmin(dists))
            ref_point, ref_room_id = all_reference_centers[nearest_idx]

            updated_map[obj_instance_masks[idx]] = ref_room_id
            new_centers_with_rooms.append((center, ref_room_id))

            # find human-readable ref info
            for c, ref_obj_name, room_id in ref_center_info:
                if c == ref_point:
                    room_name = [k for k, v in room_name_to_id.items() if v == room_id][0]
                    print(f"  → instance center ({center[0]}, {center[1]}) is closest to "
                          f"'{ref_obj_name}' center ({ref_point[0]}, {ref_point[1]}) "
                          f"→ assigned room: {room_name} (room_id={room_id})")
                    break

        return updated_map, new_centers_with_rooms

    room_map_final = room_map.copy()

    # Process categories with multiple possible rooms (in any order; if you prefer,
    # you can sort them by number of votes ascending so that fewer-vote ones go first)
    multi_vote_objects = [k for k, v in object_to_room_votes.items() if len(v) > 1]
    for obj_name in multi_vote_objects:
        area_thr = area_threshold_map.get(obj_name, DEFAULT_THRESHOLD)
        room_map_final, new_centers = assign_room_to_multi_vote_objects(room_map_final, obj_name, area_threshold=area_thr)
        multi_obj_centers_by_room[obj_name] = new_centers

    # Save final room map
    np.savetxt(OUTPUT_CSV_PATH, room_map_final, fmt="%d", delimiter=",")
    print(f"[INFO] Saved room assignment map to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
