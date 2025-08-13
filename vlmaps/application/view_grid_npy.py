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

import numpy as np
import torch
import clip


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    scene_path = data_dirs[config.scene_id]
    print(scene_path)

    vlmap = VLMap(config.map_config, data_dir=scene_path)
    vlmap.load_map(scene_path)
    visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb)

    # ========== 전체 semantic map 생성 및 저장 ==========
    print("전체 category에 대해 semantic label map 생성 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)

    categories = mp3dcat[1:-1]
    text_tokens = clip.tokenize(categories).to(device)

    with torch.no_grad():
        text_feats = model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats.to(dtype=torch.float32)

    grid_feat = torch.tensor(vlmap.grid_feat, dtype=torch.float32).to(device)
    grid_feat = grid_feat / grid_feat.norm(dim=-1, keepdim=True)

    similarity = grid_feat @ text_feats.T
    semantic_ids = torch.argmax(similarity, dim=1).cpu().numpy()

    # === 2D semantic map으로 변환
    def generate_semantic_map_2d(semantic_ids, grid_pos, resolution=0.05):
        x = grid_pos[:, 0]
        z = grid_pos[:, 2]
        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()

        width = int((x_max - x_min) / resolution) + 1
        height = int((z_max - z_min) / resolution) + 1

        semantic_map = np.zeros((height, width), dtype=np.int32) - 1

        for i in range(len(semantic_ids)):
            row = int((z[i] - z_min) / resolution)
            col = int((x[i] - x_min) / resolution)
            if 0 <= row < height and 0 <= col < width:
                semantic_map[row, col] = semantic_ids[i]

        return semantic_map

    semantic_map = generate_semantic_map_2d(semantic_ids, vlmap.grid_pos, resolution=0.05)

    unique, counts = np.unique(semantic_map, return_counts=True)
    valid_mask = unique != -1
    valid_labels = unique[valid_mask]
    valid_counts = counts[valid_mask]

    if len(valid_labels) > 0:
        dominant_label = valid_labels[np.argmax(valid_counts)]
        print(f"가장 많이 나온 label: {dominant_label} ({mp3dcat[dominant_label + 1]})")
    
        semantic_map[semantic_map == dominant_label] = -1
    else:
        print("마스킹할 유효한 semantic label이 없습니다.")
    # === 저장
    output_path = scene_path / "map" / "grid_1_clip.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, semantic_map)
    print(f"grid_1_clip.npy 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
