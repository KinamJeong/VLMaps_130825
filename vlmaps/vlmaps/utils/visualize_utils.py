import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

#index_map_semantic.py
from collections import defaultdict, Counter

def visualize_rgb_map_3d(pc: np.ndarray, rgb: np.ndarray):
    grid_rgb = rgb / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
    o3d.visualization.draw_geometries([pcd])


def get_heatmap_from_mask_3d(
    pc: np.ndarray, mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01
) -> np.ndarray:
    target_pc = pc[mask, :]
    other_ids = np.where(mask == 0)[0]
    other_pc = pc[other_ids, :]

    target_sim = np.ones((target_pc.shape[0], 1))
    other_sim = np.zeros((other_pc.shape[0], 1))
    pbar = tqdm(other_pc, desc="Computing heat", total=other_pc.shape[0])
    for other_p_i, p in enumerate(pbar):
        dist = np.linalg.norm(target_pc - p, axis=1) / cell_size
        min_dist_i = np.argmin(dist)
        min_dist = dist[min_dist_i]
        other_sim[other_p_i] = np.clip(1 - min_dist * decay_rate, 0, 1)

    new_pc = pc.copy()
    heatmap = np.ones((new_pc.shape[0], 1), dtype=np.float32)
    for s_i, s in enumerate(other_sim):
        heatmap[other_ids[s_i]] = s
    return heatmap.flatten()


def visualize_masked_map_3d(pc: np.ndarray, mask: np.ndarray, rgb: np.ndarray, transparency: float = 0.5):
    heatmap = mask.astype(np.float16)
    visualize_heatmap_3d(pc, heatmap, rgb, transparency)


def visualize_heatmap_3d(pc: np.ndarray, heatmap: np.ndarray, rgb: np.ndarray, transparency: float = 0.5):
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat.reshape(-1, 3)[:, ::-1].astype(np.float32)
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    visualize_rgb_map_3d(pc, heat_rgb)


def pool_3d_label_to_2d(mask_3d: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    mask_2d = np.zeros((gs, gs), dtype=bool)

    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        mask_2d[row, col] = mask_3d[i] or mask_2d[row, col]

    return mask_2d

# index_map_semantic.py
def pool_3d_index_to_2d(max_ids: np.ndarray, grid_pos: np.ndarray, gs: int,h_min: int=10) -> np.ndarray:
    semantic_map_2d = -1 * np.ones((gs, gs), dtype=int)
    height_map = -100 * np.ones((gs, gs), dtype=int)

    #특정 높이 이상의 point중 가장 높은 z값의 semantic 정보만 사용 / 10 = 0.5m , 20 = 1m, 40 = 2m
    for i in range(len(max_ids)):
        r, c, h = grid_pos[i]
        if 0 <= r < gs and 0 <= c < gs and h >= h_min:
            if h > height_map[r, c]:
                semantic_map_2d[r, c] = max_ids[i]
                height_map[r, c] = h

    #semantic_map_2d.csv 출력
    save_path = f"semantic_map_RP_2d_high_h{h_min}.csv"

    # CSV로 저장
    np.savetxt(save_path, semantic_map_2d, fmt='%d', delimiter=",")
    print(f"[INFO] semantic_map_2d saved to: {save_path} (min height: h >= {h_min})")

    return semantic_map_2d

def pool_3d_index_to_2d_majority(max_ids: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:

    semantic_votes = defaultdict(list)
    for i in range(len(max_ids)):
        r, c, _ = grid_pos[i]
        if 0 <= r < gs and 0 <= c < gs:
            semantic_votes[(r, c)].append(max_ids[i])
    
    semantic_map_2d = -1 * np.ones((gs, gs), dtype=int)
    for (r, c), labels in semantic_votes.items():
        if 6 in labels:
            semantic_map_2d[r,c]=6 #override with tv_monitor
        elif 37 in labels:
            semantic_map_2d[r, c] = 37 #override with mirror
        else:
            most_common_label = Counter(labels).most_common(1)[0][0]
            semantic_map_2d[r, c] = most_common_label

    save_path = f"semantic_map_17D_2d_majority_tv_mirror_override.csv"
    np.savetxt(save_path, semantic_map_2d, fmt='%d', delimiter=",")
    print(f"[INFO] semantic_map_2d (majority voting with tv_override) saved to: {save_path}")

    return semantic_map_2d

def pool_3d_rgb_to_2d(rgb: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    rgb_2d = np.zeros((gs, gs, 3), dtype=np.uint8)
    height = -100 * np.ones((gs, gs), dtype=np.int32)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        if h > height[row, col]:
            rgb_2d[row, col] = rgb[i]

    return rgb_2d


def get_heatmap_from_mask_2d(mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01) -> np.ndarray:
    dists = distance_transform_edt(mask == 0) / cell_size
    tmp = np.ones_like(dists) - (dists * decay_rate)
    heatmap = np.where(tmp < 0, np.zeros_like(tmp), tmp)

    return heatmap


def visualize_rgb_map_2d(rgb: np.ndarray):
    """visualize rgb image

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
    """
    rgb = rgb.astype(np.uint8)
    bgr = rgb[:, :, ::-1]
    cv2.imshow("rgb map", bgr)
    cv2.waitKey(0)


def visualize_heatmap_2d(rgb: np.ndarray, heatmap: np.ndarray, transparency: float = 0.5):
    """visualize heatmap

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        heatmap (np.ndarray): (gs, gs) element range [0, 1] np.float32
    """
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat[:, :, ::-1].astype(np.float32)  # convert to RGB
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    visualize_rgb_map_2d(heat_rgb)


def visualize_masked_map_2d(rgb: np.ndarray, mask: np.ndarray):
    """visualize masked map

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        mask (np.ndarray): (gs, gs) element range [0, 1] np.uint8
    """
    visualize_heatmap_2d(rgb, mask.astype(np.float32))

# index_map_semantic.py
def visualize_semantic_map_2d(
        rgb: np.ndarray,
        semantic_map: np.ndarray,
        transparency: float = 0,
        num_classes: int = 40
        ):
    
    H, W = semantic_map.shape

    #색상 매핑 딕셔너리
    class_color_map = {
    0: [255, 0, 0],      # table
    1: [0, 255, 0],      # sofa
    2: [0, 0, 255],      # bed
    3: [255, 255, 0],    # sink
    4: [255, 0, 255],    # chair
    5: [0, 255, 255],    # toilet
    6: [255, 128, 0],    # tv_monitor
    7: [128, 0, 255],    # wall
    8: [135, 206, 235],  # window
    15: [128,128,0],     #shelving
    17: [192, 192, 192],  # cabinet
    32: [255, 215, 0],   #refrigerator 
    34: [0, 128, 128],   # microwave
    26: [255, 69, 0],    # counter
    14: [0, 100, 0],     # tree
    16: [70, 130, 180],  # door
    17: [169, 169, 169], # cabinet
    18: [255, 20, 147],  # curtain
    26: [0, 0, 139],     # counter
    29: [0, 128, 128],   # stove
    31: [160, 82, 45],   # blinds
    22 : [45,82,160],    #desktop_computer 
    38: [64, 64, 64],   # objects
}

    #other → grayish olive
    default_color = np.array([255, 255, 255], dtype=np.uint8) 
   
    semantic_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in class_color_map.items():
        semantic_rgb[semantic_map == cls_id] = np.array(color, dtype=np.uint8)
    
    # default 처리: class_color_map에 없는 모든 클래스
    mask = np.isin(semantic_map, list(class_color_map.keys()), invert=True)
    semantic_rgb[mask] = default_color
    
    blended = (semantic_rgb.astype(np.float32) * transparency +
               rgb.astype(np.float32) * (1 - transparency)).astype(np.uint8)
    
    visualize_rgb_map_2d(blended)

# index_map_semantic.py
def visualize_semantic_map_3d(
        pc: np.ndarray, 
        max_ids: np.ndarray, 
        rgb: np.ndarray, 
        transparency: float = 0.5
        ):
    
     #색상 매핑 딕셔너리
    class_color_map = {
    0: [255, 0, 0],      # table
    1: [0, 255, 0],      # sofa
    2: [0, 0, 255],      # bed
    3: [255, 255, 0],    # sink
    4: [255, 0, 255],    # chair
    5: [0, 255, 255],    # toilet
    6: [255, 128, 0],    # tv_monitor
    7: [128, 0, 255],    # wall
    8: [135, 206, 235],  # window
    15: [128,128,0],     #shelving
    17: [192, 192, 192],  # cabinet
    32: [255, 215, 0],   #refrigerator 
    34: [0, 128, 128],   # microwave
    26: [255, 69, 0],    # counter
    14: [0, 100, 0],     # tree
    16: [70, 130, 180],  # door
    17: [169, 169, 169], # cabinet
    18: [255, 20, 147],  # curtain
    26: [0, 0, 139],     # counter
    29: [0, 128, 128],   # stove
    31: [160, 82, 45],   # blinds
    22 : [45,82,160],    #desktop_computer 
    38: [64, 64, 64],   # objects
}

    default_color = np.array([64, 64, 0], dtype=np.uint8)

    semantic_rgb = np.zeros_like(rgb, dtype=np.uint8)
    for cls_id in np.unique(max_ids):
        color = class_color_map.get(int(cls_id), default_color)
        semantic_rgb[max_ids == cls_id] = color

    blended = (semantic_rgb.astype(np.float32) * transparency +
               rgb.astype(np.float32) * (1 - transparency)).astype(np.uint8)
    
    visualize_rgb_map_3d(pc, blended)