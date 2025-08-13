from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)

import cv2
import numpy as np

def remove_noise_from_mask(mask_2d: np.ndarray, min_area: int = 1000) -> np.ndarray:
    # Step 1: bool → uint8 변환
    mask = mask_2d.astype(np.uint8) * 255
    
    # Step 2: contour 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 3: 넓은 contour만 남기기
    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return clean_mask.astype(bool)

def get_object_center(vlmap, category: str, gs: int, min_area: int) -> tuple:
    """주어진 카테고리 이름에 대해 중심 좌표를 반환"""
    mask = vlmap.index_map(category, with_init_cat=True)
    mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, gs)
    clean_mask = remove_noise_from_mask(mask_2d, min_area=min_area)
    coords = np.argwhere(clean_mask)
    if coords.size == 0:
        raise ValueError(f"No valid region found for category: {category}")
    center = coords.mean(axis=0)
    return center, clean_mask

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    print(data_dirs[config.scene_id])
    vlmap = VLMap(config.map_config, data_dir=data_dirs[config.scene_id])
    vlmap.load_map(data_dirs[config.scene_id])
    visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb)

    cat1 = input("Enter first category (e.g., 'bed'): ")
    cat2 = input("Enter second category (e.g., 'chair'): ")
    # cat = "chair"

    vlmap._init_clip()
    print("considering categories: ")
    print(mp3dcat[1:-1])
    if config.init_categories:
        vlmap.init_categories(mp3dcat[1:-1])

    min_area1 = 1000 if cat1 == "bed" else 200
    min_area2 = 100 if cat2 == "chair" else 200
    center1, mask1 = get_object_center(vlmap, cat1, config.params.gs, min_area1)
    center2, mask2 = get_object_center(vlmap, cat2, config.params.gs, min_area2)

    midpoint = (center1 + center2) / 2.0
    print(f"\ncenter of {cat1}: {center1}")
    print(f"center of {cat2}: {center2}")
    print(f"midpoint between {cat1} and {cat2}: {midpoint}")
    
    rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)

    midpoint_int = tuple(np.round(midpoint).astype(int)[::-1])
    rgb_vis = rgb_2d.copy()
    cv2.circle(rgb_vis, midpoint_int, radius=5, color=(0, 0, 255), thickness=-1)
    cv2.circle(rgb_vis, tuple(np.round(center1)[::-1].astype(int)), radius=5, color=(0, 255, 0), thickness=-1)  
    cv2.circle(rgb_vis, tuple(np.round(center2)[::-1].astype(int)), radius=5, color=(255, 0, 0), thickness=-1)
    cv2.imshow("RGB map with midpoint", rgb_vis[:, :, ::-1])  # RGB → BGR 변환
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #visualize_masked_map_2d(rgb_2d, mask1 | mask2)

if __name__ == "__main__":
    main()
