import os
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import hydra

from vlmaps.task.habitat_object_nav_task import HabitatObjectNavigationTask
from vlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from vlmaps.utils.llm_utils import parse_object_goal_instruction
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.mapping_utils import cvt_pose_vec2tf

import time #자연어 처리 시간 측정

#랜덤 위치 설정
import random

start_poses = [
    [380, 380, 1.0],
    [385, 425, 1.0],
    [340, 470, 1.0],
    [430, 470, 1.0]
]

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    data_dir = Path(config.data_paths.vlmaps_data_dir) / "vlmaps_dataset"
    robot = HabitatLanguageRobot(config)
    object_nav_task = HabitatObjectNavigationTask(config)
    object_nav_task.reset_metrics()
    scene_ids = []
    if isinstance(config.scene_id, int):
        scene_ids.append(config.scene_id)
    else:
        scene_ids = config.scene_id

    for scene_i, scene_id in enumerate(scene_ids):
        robot.setup_scene(scene_id)
        robot.map.init_categories(mp3dcat.copy())

        row, col, yaw_deg = random.choice(start_poses)
        robot.vlmaps_dataloader.from_full_map_pose(row, col, yaw_deg) 
        hab_tf = robot.vlmaps_dataloader.to_habitat_tf()
        robot.set_agent_state(hab_tf)
        
        object_nav_task.setup_scene(robot.vlmaps_dataloader)
        object_nav_task.load_task()


        for task_id in range(len(object_nav_task.task_dict)):
            object_nav_task.setup_task(task_id)

            row, col, yaw_deg = random.choice(start_poses)
            robot.vlmaps_dataloader.from_full_map_pose(row, col, yaw_deg) 
            hab_tf = robot.vlmaps_dataloader.to_habitat_tf()
            robot.set_agent_state(hab_tf)
        

            start_time = time.time()
            object_categories = parse_object_goal_instruction(object_nav_task.instruction)
            elapsed = time.time() - start_time
            print(f"[TIME] parse_object_goal_instruction took {elapsed:.3f} seconds")
            with open("/home/ivl1/00_workspace/evaluation/vlmaps_time_result_log.txt", "a") as f:
                   f.write(f"Time : {elapsed:.3f} (parsing)\n")

            print(f"instruction: {object_nav_task.instruction}")
            robot.empty_recorded_actions()
            #robot.set_agent_state(object_nav_task.init_hab_tf)
        
            for cat_i, cat in enumerate(object_categories):
                print(f"Navigating to category {cat}")
                actions_list = robot.move_to_object_origin(cat)
                
            
            with open("/home/ivl1/00_workspace/evaluation/vlmaps_category_result_log.txt", "a") as f:
                f.write("\n")
            with open("/home/ivl1/00_workspace/evaluation/vlmaps_navigation_result_log.txt", "a") as f:
                f.write("\n\n")
            with open("/home/ivl1/00_workspace/evaluation/vlmaps_time_result_log.txt", "a") as f:
                f.write("\n\n")
            #print(f"[DEBUG] Length of actions_list: {len(actions_list)}")

            recorded_actions_list = robot.get_recorded_actions()
            #robot.set_agent_state(object_nav_task.init_hab_tf) #이게 초기화 버튼인건가? 원래는 주석처리 되어있었음
            robot.set_agent_state(hab_tf) #내가 설정한 초기 위치로 다시 돌아가기?
            
            #print(f"[DEBUG] evaluate_object: Length of recorded_actions_list: {len(recorded_actions_list)}")

            #for action in recorded_actions_list:
            #    object_nav_task.test_step(robot.sim, action, vis=config.nav.vis)
                
            #print(f"[DEBUG] : test_step")

            #save_dir = robot.vlmaps_dataloader.data_dir / (config.map_config.map_type + "_obj_nav_results")
            #os.makedirs(save_dir, exist_ok=True)
            #save_path = save_dir / f"{task_id:02}.json"
            #object_nav_task.save_single_task_metric(save_path)

            #robot.set_agent_state(hab_tf)


if __name__ == "__main__":
    main()
