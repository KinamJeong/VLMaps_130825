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

    pool_3d_index_to_2d, #index_map_semantic.py
    pool_3d_index_to_2d_majority,
    visualize_semantic_map_2d, #index_map_semantic.py
    visualize_semantic_map_3d, #index_map_semantic.py
   
)


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


    #cat = input("What is your interested category in this scene?")
    # cat = "chair"

    vlmap._init_clip()
    print("considering categories: ")
    print(mp3dcat[1:-1])
    if config.init_categories:
        print("init_categories")
        vlmap.init_categories(mp3dcat[1:-1])
        max_ids = vlmap.total_index_map()
    else:
        max_ids = vlmap.total_index_map()

    #2D Top-down segmentation map 만들 때,
    #Z축 가장 위 쪽 픽셀 인덱스 선택 시 False, 가장 많은 픽셀 인덱스 선택 시 True
    majority = True

    if config.index_2d:
        if majority:
            semantic_map_2d = pool_3d_index_to_2d_majority(max_ids,vlmap.grid_pos,config.params.gs)
        else:
            semantic_map_2d = pool_3d_index_to_2d(max_ids,vlmap.grid_pos,config.params.gs)
        
        rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)
        visualize_semantic_map_2d(rgb_2d,semantic_map_2d,transparency=1, num_classes=40)

        # 방 분리 맵 만드는 부분 : semantic_map_2d 가지고 만들기
        
        #heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
        #visualize_heatmap_2d(rgb_2d, heatmap)
    else:
        visualize_semantic_map_3d(pc=vlmap.grid_pos, max_ids=max_ids, rgb=vlmap.grid_rgb, transparency=1.0)
        #visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb)
        #heatmap = get_heatmap_from_mask_3d(
        #    vlmap.grid_pos, mask, cell_size=config.params.cs, decay_rate=config.decay_rate
        # )
        #visualize_heatmap_3d(vlmap.grid_pos, heatmap, vlmap.grid_rgb)


if __name__ == "__main__":
    main()
