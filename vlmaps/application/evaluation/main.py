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
#main.py
from google_flan_t5_base import process_commands_from_file

#랜덤 위치 설정
import random

start_poses = [
    [380, 380, 1.0],
    #[385, 425, 1.0],
    #[340, 470, 1.0],
    #[430, 470, 1.0]
]

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    main_start_time = time.time()
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
        
        # 방 상태 정보 가져오기
        #robot.map.load_room_labels("room_labels_0717_3.csv")
        robot.map.load_room_labels("room_assignment_all_objects.csv")

        row, col, yaw_deg = random.choice(start_poses) # 1000 : 465,465, 0.0 / 2000 : 300, 320 , 0.0
        robot.vlmaps_dataloader.from_full_map_pose(row, col, yaw_deg)
        hab_tf = robot.vlmaps_dataloader.to_habitat_tf()
        robot.set_agent_state(hab_tf)
        
        object_nav_task.setup_scene(robot.vlmaps_dataloader)
        object_nav_task.load_task()

        main_elapsed = time.time() - main_start_time
        print(f"[TIME] From start of main() to Scene Setup: {main_elapsed:.3f} seconds")
        # 자연어 처리 결과 txt 파일에 저장
        #save_log_path = "/home/ivl1/00_workspace/evaluation/inference_result_log.txt" 

        start_llm_time = time.time()
        # llm_sentence.txt에서 문장 가져오기
        file_path = "/home/ivl1/chatbot_server/chatlogs_eng/llm_sentence.txt"
        # 자연어 처리 : 객체, 장소 추출
        
        place, object_categories= process_commands_from_file(file_path)
        llm_elapsed = time.time() - start_llm_time
        print(f"[TIME] parse_object_goal_instruction took {llm_elapsed:.3f} seconds")
        
        #print("\nFinal Lists_main.py:")
        #print("Locations:", place)
        #print("Objects:  ", object_categories)

        #with open(save_log_path, "w", encoding="utf-8") as f:
        #    f.write(f"[TIME] parse_object_goal_instruction took {elapsed:.3f} seconds\n\n")
        #    f.write("Final Lists_main.py:\n")
        #    f.write(f"Locations: {place}\n")
        #    f.write(f"Objects:   {object_categories}\n")

        for task_id in range(len(object_nav_task.task_dict)):
            start_task_setup_time = time.time()
            object_nav_task.setup_task(task_id)
            #print(f"instruction: {object_nav_task.instruction}")
            # llm_sentence의 문장을 읽어 이를 자연어 파싱에 사용
            #with open("/home/ivl1/chatbot_server/llm_sentence.txt","r",encoding="utf-8") as f:
            #    instruction_from_file = f.read().strip()
            
            # 자연어 명령어 파싱 시간 측정
            
            # llm_sentence의 문장을 읽어 이를 자연어 파싱에 사용

            #toy example로 "go to sink" 로 하고, 
            #object_categories = parse_object_goal_instruction(instruction_from_file)
           
            robot.empty_recorded_actions()
            #robot.set_agent_state(object_nav_task.init_hab_tf)

            task_setup_elapsed = time.time() - start_task_setup_time
            print(f"[TIME] From start of main() to just before navigating: {task_setup_elapsed:.3f} seconds")
            for cat,pl in zip(object_categories,place):   
                print(f"Navigating to category {cat}")
                actions_list = robot.move_to_object(cat,place=pl)

            with open("/home/ivl1/00_workspace/evaluation/navigation_result_log.txt", "a") as f:
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


if __name__ == "__main__":
    main()
