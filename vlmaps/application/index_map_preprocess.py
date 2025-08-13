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
    cat = input("What is your interested category in this scene?")
    # cat = "chair"

    vlmap._init_clip()
    print("considering categories: ")
    print(mp3dcat[1:-1])
    if config.init_categories:
        vlmap.init_categories(mp3dcat[1:-1])
        mask = vlmap.index_map(cat, with_init_cat=True)
    else:
        mask = vlmap.index_map(cat, with_init_cat=False)

    if config.index_2d:
        mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, config.params.gs)
        rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)

        if cat == 'chair':
            clean_mask_2d = remove_noise_from_mask(mask_2d, min_area=100)
        elif cat == "bed":
            clean_mask_2d = remove_noise_from_mask(mask_2d, min_area=1000)
        else :
            clean_mask_2d = remove_noise_from_mask(mask_2d, min_area=200)

        visualize_masked_map_2d(rgb_2d, clean_mask_2d)
        heatmap = get_heatmap_from_mask_2d(clean_mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
        visualize_heatmap_2d(rgb_2d, heatmap)
        coords = np.argwhere(clean_mask_2d)
        center = coords.mean(axis=0)
        print("center of "+cat+" : ")
        print(center)
    else:
        visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb)
        heatmap = get_heatmap_from_mask_3d(
            vlmap.grid_pos, mask, cell_size=config.params.cs, decay_rate=config.decay_rate
        )
        visualize_heatmap_3d(vlmap.grid_pos, heatmap, vlmap.grid_rgb)


if __name__ == "__main__":
    main()
