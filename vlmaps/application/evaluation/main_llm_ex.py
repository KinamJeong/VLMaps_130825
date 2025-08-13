import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import hydra
import time
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

from vlmaps.task.habitat_object_nav_task import HabitatObjectNavigationTask
from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.mapping_utils import cvt_pose_vec2tf

# ====== Load FLAN-T5 model once ======
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

location_list = ["kitchen", "bathroom", "bedroom", "living room", "study room"]

object2locations = {
    "bed": ["bedroom"], "sofa": ["living room"], "stove": ["kitchen"], "counter": ["kitchen"],
    "microwave": ["kitchen"], "refrigerator": ["kitchen"], "toilet": ["bathroom"],
    "desktop_computer": ["study room"], "table": ["living room", "study room"],
    "sink": ["bathroom", "kitchen"], "window": ["kitchen", "bedroom"],
    "shelving": ["study room", "kitchen"], "curtain": ["bedroom", "bathroom"],
    "chair": ["living room", "study room"], "tv_monitor": ["living room", "bedroom", "study room"],
    "cabinet": ["study room", "kitchen", "bedroom"]
}


def infer_location(command: str, top_k: int = 3):
    prompt = (
        f"Choose the best room for this task: {command}\n"
        f"Options: kitchen, bathroom, bedroom, living room, study room.\n"
        f"Answer with the room name only."
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=16)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()
    prediction_clean = prediction.replace('.', '').replace(',', '').strip()
    matches = [loc for loc in location_list if loc in prediction_clean]
    best_location = matches[0] if matches else "unknown"

    top_candidates = []
    with torch.no_grad():
        pred_emb = model.encoder(**tokenizer(prediction, return_tensors="pt")).last_hidden_state.mean(dim=1)
        for loc in location_list:
            loc_emb = model.encoder(**tokenizer(loc, return_tensors="pt")).last_hidden_state.mean(dim=1)
            sim = F.cosine_similarity(pred_emb, loc_emb).item()
            top_candidates.append((loc, sim))
    top_candidates.sort(key=lambda x: x[1], reverse=True)
    return best_location, top_candidates[:top_k]


def get_most_relevant_object(command: str, location: str, top_k: int = 3):
    candidate_objects = [obj for obj, locs in object2locations.items() if location in locs]
    if not candidate_objects:
        return "no relevant object", []
    with torch.no_grad():
        cmd_emb = model.encoder(**tokenizer(command, return_tensors="pt")).last_hidden_state.mean(dim=1)
        similarities = []
        for obj in candidate_objects:
            obj_emb = model.encoder(**tokenizer(obj, return_tensors="pt")).last_hidden_state.mean(dim=1)
            sim = F.cosine_similarity(cmd_emb, obj_emb).item()
            similarities.append((obj, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[0][0], similarities[:top_k]


def process_commands_from_file(file_path):
    locations, objects = [], []
    with open(file_path, 'r') as f:
        commands = [line.strip() for line in f if line.strip()]

    for cmd in commands:
        loc, loc_candidates = infer_location(cmd)
        obj, obj_candidates = get_most_relevant_object(cmd, loc)
        print(f"\n🔎 Command: {cmd}")
        print(f"→ Inferred Location: {loc}")
        print("→ Top Location Candidates:")
        for name, score in loc_candidates:
            print(f"   - {name}: {score:.4f}")
        print(f"→ Inferred Object: {obj}")
        print("→ Top Object Candidates:")
        for name, score in obj_candidates:
            print(f"   - {name}: {score:.4f}")
        locations.append(loc)
        objects.append(obj)
    return locations, objects


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    robot = HabitatLanguageRobot(config)
    object_nav_task = HabitatObjectNavigationTask(config)
    object_nav_task.reset_metrics()

    scene_ids = [config.scene_id] if isinstance(config.scene_id, int) else config.scene_id

    for scene_id in scene_ids:
        robot.setup_scene(scene_id)
        robot.map.init_categories(mp3dcat.copy())
        robot.map.load_room_labels("room_assignment_all_objects.csv")

        row, col, yaw_deg = 385, 425, 1.0
        robot.vlmaps_dataloader.from_full_map_pose(row, col, yaw_deg)
        hab_tf = robot.vlmaps_dataloader.to_habitat_tf()
        robot.set_agent_state(hab_tf)

        object_nav_task.setup_scene(robot.vlmaps_dataloader)
        object_nav_task.load_task()

        for task_id in range(len(object_nav_task.task_dict)):
            object_nav_task.setup_task(task_id)
            robot.empty_recorded_actions()

        
            locs, objs = process_commands_from_file("llm_sentence.txt")
            place = locs[0]
            object_categories = [objs[0]]

            for cat in object_categories:
                print(f"Navigating to category {cat} in {place}")
                actions_list = robot.move_to_object(cat, place=place)

            robot.set_agent_state(hab_tf)

            recorded_actions_list = robot.get_recorded_actions()
            for action in recorded_actions_list:
                object_nav_task.test_step(robot.sim, action, vis=config.nav.vis)

            save_dir = robot.vlmaps_dataloader.data_dir / (config.map_config.map_type + "_obj_nav_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = save_dir / f"{task_id:02}.json"
            object_nav_task.save_single_task_metric(save_path)


if __name__ == "__main__":
    main()
