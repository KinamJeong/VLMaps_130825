import h5py
import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d

# === 1. Load VLMap data ===
h5_path = "vlmaps.h5df"
with h5py.File(h5_path, "r") as f:
    grid_feat = f["grid_feat"][:]      # (N, D)
    grid_pos = f["grid_pos"][:]        # (N, 3)
    grid_rgb = f["grid_rgb"][:]        # (N, 3)

# === 2. Load CLIP model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
clip_feat_dim = 512

# === 3. Preprocess category list ===
categories = mp3dcat[1:-1]  # "other" 제외
text_tokens = clip.tokenize(categories).to(device)

with torch.no_grad():
    text_feats = model.encode_text(text_tokens)  # (C, D)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.to(dtype=torch.float32)

# === 4. Normalize voxel features ===
grid_feat_torch = torch.tensor(grid_feat, dtype=torch.float32).to(device)
grid_feat_torch = grid_feat_torch / grid_feat_torch.norm(dim=-1, keepdim=True)

# === 5. Calculate similarity & assign labels ===
similarity = grid_feat_torch @ text_feats.T  # (N, C)
semantic_ids = torch.argmax(similarity, dim=1).cpu().numpy()  # (N,)

print("Semantic labels 생성 완료:", semantic_ids.shape)

# (Optional) 저장
np.save("semantic_ids.npy", semantic_ids)
