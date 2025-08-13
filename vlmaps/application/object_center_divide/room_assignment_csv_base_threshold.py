
import json
import numpy as np
from scipy.ndimage import label, center_of_mass
from typing import Dict, List

# -------------------------
# File paths (adjust if needed)
# -------------------------
SEMANTIC_MAP_PATH = "semantic_map_RP_2d_majority_tv_override_0717.csv"
OUTPUT_CSV_PATH   = "room_assignment_RP_all_objects_from_csv_base.csv"
OUTPUT_THRESH_JSON = "csv_base_thresholds.json"

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
# CSV-based BASE thresholds (computed from this scene)
# -------------------------
MIN_PX, MAX_PX = 5, 200     # clamp to avoid pathological values
DEFAULT_THRESHOLD = 50       # fallback if no instances found
PERCENTILE = 10              # use lower percentile for robustness to large outliers
MIN_INSTANCES = 3            # require at least N instances to trust the percentile

def compute_base_thresholds_from_csv(
    semantic_map: np.ndarray,
    mp3dcat_trimmed: List[str],
    percentile: int = PERCENTILE,
    min_instances: int = MIN_INSTANCES,
    default_threshold: int = DEFAULT_THRESHOLD,
) -> Dict[str, int]:
    """
    Compute a per-category base threshold (in pixels) using connected-component
    areas from the CSV (semantic_map). Returns a dict {category_name: threshold_px}.
    """
    thresholds = {}
    for obj_name in mp3dcat_trimmed:
        obj_id = mp3dcat_trimmed.index(obj_name)
        mask = (semantic_map == obj_id).astype(np.uint8)
        if mask.sum() == 0:
            # No instances in this scene
            thresholds[obj_name] = default_threshold
            continue

        labeled, num = label(mask)
        areas = []
        for i in range(1, num + 1):
            area = int((labeled == i).sum())
            if area > 0:
                areas.append(area)

        if len(areas) >= min_instances:
            p = int(np.percentile(areas, percentile))
            thr = max(default_threshold, p)  # never go below default
        elif len(areas) > 0:
            # Use median when instances are few but non-zero
            p = int(np.median(areas))
            thr = max(default_threshold, p)
        else:
            thr = default_threshold

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

    # 1) Compute CSV-based BASE thresholds (per-category)
    base_threshold_map = compute_base_thresholds_from_csv(
        semantic_map,
        mp3dcat_trimmed,
        percentile=PERCENTILE,
        min_instances=MIN_INSTANCES,
        default_threshold=DEFAULT_THRESHOLD,
    )

    # Save thresholds for record/debugging
    with open(OUTPUT_THRESH_JSON, "w") as f:
        json.dump(base_threshold_map, f, indent=2)
    print(f"[INFO] Saved CSV-based base thresholds to: {OUTPUT_THRESH_JSON}")

    # 2) Build room map using these thresholds
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
        room_map[semantic_map == obj_id] = room_id

        # Use CSV-based base threshold here
        threshold = base_threshold_map.get(obj_name, DEFAULT_THRESHOLD)
        mask = (semantic_map == obj_id).astype(np.uint8)
        labeled, num = label(mask)

        centers = []
        count = 0
        for i in range(1, num + 1):
            region_mask = (labeled == i)
            area = int(region_mask.sum())
            if area < threshold:
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
            area = int(instance_mask.sum())
            if area < area_threshold:
                continue
            center = center_of_mass(instance_mask)
            center_int = tuple(map(int, center))
            obj_centers.append(center_int)
            obj_instance_masks.append(instance_mask)
            print(f"  - instance {i}: center (row={center_int[0]}, col={center_int[1]}), area={area}")

        # Build reference anchors
        all_reference_centers = []  # (center_xy, room_id)
        ref_center_info = []        # (center_xy, ref_obj_name, room_id)

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

        if not all_reference_centers:
            print("[WARN] No reference anchors yet; skipping assignment for this category.")
            return updated_map, new_centers_with_rooms

        for idx, center in enumerate(obj_centers):
            # nearest anchor (Euclidean in pixel space)
            point = np.array([center])
            dists = [np.linalg.norm(point - np.array(x[0])) for x in all_reference_centers]
            nearest_idx = int(np.argmin(dists))
            ref_point, ref_room_id = all_reference_centers[nearest_idx]

            updated_map[obj_instance_masks[idx]] = ref_room_id
            new_centers_with_rooms.append((center, ref_room_id))

            # print a readable log
            for c, ref_obj_name, room_id in ref_center_info:
                if c == ref_point:
                    room_name = [k for k, v in room_name_to_id.items() if v == room_id][0]
                    print(f"  → instance center ({center[0]}, {center[1]}) closest to "
                          f"'{ref_obj_name}' center ({ref_point[0]}, {ref_point[1]}) "
                          f"→ assigned: {room_name} (room_id={room_id})")
                    break

        return updated_map, new_centers_with_rooms

    room_map_final = room_map.copy()

    multi_vote_objects = [k for k, v in object_to_room_votes.items() if len(v) > 1]
    for obj_name in multi_vote_objects:
        area_thr = base_threshold_map.get(obj_name, DEFAULT_THRESHOLD)
        room_map_final, new_centers = assign_room_to_multi_vote_objects(room_map_final, obj_name, area_threshold=area_thr)
        multi_obj_centers_by_room[obj_name] = new_centers

    # Save final room map
    np.savetxt(OUTPUT_CSV_PATH, room_map_final, fmt="%d", delimiter=",")
    print(f"[INFO] Saved room assignment map to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
