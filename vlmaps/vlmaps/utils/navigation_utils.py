import numpy as np
import cv2
from scipy.spatial.distance import cdist
import pyvisgraph as vg
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict

from scipy.ndimage import binary_closing

#main.py
from collections import deque , defaultdict

def get_segment_islands_pos(segment_map, label_id, detect_internal_contours=False):
    mask = segment_map == label_id
    mask = mask.astype(np.uint8)
    #좁은 통로 닫기
    #mask = binary_closing(mask, structure=np.ones((3, 3)), iterations=1).astype(np.uint8)

    detect_type = cv2.RETR_EXTERNAL
    if detect_internal_contours:
        detect_type = cv2.RETR_TREE

    contours, hierarchy = cv2.findContours(mask, detect_type, cv2.CHAIN_APPROX_SIMPLE)
    # convert contours back to numpy index order
    contours_list = []
    for contour in contours:
        tmp = contour.reshape((-1, 2))
        tmp_1 = np.stack([tmp[:, 1], tmp[:, 0]], axis=1)
        contours_list.append(tmp_1)

    centers_list = []
    bbox_list = []
    for c in contours_list:
        xmin = np.min(c[:, 0])
        xmax = np.max(c[:, 0])
        ymin = np.min(c[:, 1])
        ymax = np.max(c[:, 1])
        bbox_list.append([xmin, xmax, ymin, ymax])

        centers_list.append([(xmin + xmax) / 2, (ymin + ymax) / 2])

    return contours_list, centers_list, bbox_list, hierarchy


def find_closest_points_between_two_contours(obs_map, contour_a, contour_b):
    a = np.zeros_like(obs_map, dtype=np.uint8)
    b = np.zeros_like(obs_map, dtype=np.uint8)
    cv2.drawContours(a, [contour_a[:, [1, 0]]], 0, 255, 1)
    cv2.drawContours(b, [contour_b[:, [1, 0]]], 0, 255, 1)
    rows_a, cols_a = np.where(a == 255)
    rows_b, cols_b = np.where(b == 255)
    pts_a = np.concatenate([rows_a.reshape((-1, 1)), cols_a.reshape((-1, 1))], axis=1)
    pts_b = np.concatenate([rows_b.reshape((-1, 1)), cols_b.reshape((-1, 1))], axis=1)
    dists = cdist(pts_a, pts_b)
    id = np.argmin(dists)
    ida, idb = np.unravel_index(id, dists.shape)
    return [rows_a[ida], cols_a[ida]], [rows_b[idb], cols_b[idb]]


def point_in_contours(obs_map, contours_list, point):
    """
    obs_map: np.ndarray, 1 free, 0 occupied
    contours_list: a list of cv2 contours [[(col1, row1), (col2, row2), ...], ...]
    point: (row, col)
    """
    row, col = int(point[0]), int(point[1])
    ids = []
    print("contours num: ", len(contours_list))
    for con_i, contour in enumerate(contours_list):
        contour_cv2 = contour[:, [1, 0]]
        con_mask = np.zeros_like(obs_map, dtype=np.uint8)
        cv2.drawContours(con_mask, [contour_cv2], 0, 255, -1)
        # con_mask_copy = con_mask.copy()
        # cv2.circle(con_mask_copy, (col, row), 10, 0, 3)
        # cv2.imshow("contour_mask", con_mask_copy)
        # cv2.waitKey()
        if con_mask[row, col] == 255:
            ids.append(con_i)

    return ids
def is_near_crop_boundary(contour, rmin, rmax, cmin, cmax, margin=1):
     return np.any(
        (contour[:, 0] <= rmin + margin) |
        (contour[:, 0] >= rmax - margin) |
        (contour[:, 1] <= cmin + margin) |
        (contour[:, 1] >= cmax - margin)
    )

def build_visgraph_with_obs_map(obs_map, use_internal_contour=False, internal_point=None, vis=False):
    obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
    obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
    #1. 기본 장애물 맵 시각화
    if vis:
       cv2.imshow("obs_first", obs_map_vis)
       cv2.waitKey(0)
       #cv2.destroyWindow("obs")

    contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
        obs_map, 0, detect_internal_contours=use_internal_contour
    )

    #2. 내부 contour 처리 시 연결 선 시각화
    if use_internal_contour:
        ids = point_in_contours(obs_map, contours_list, internal_point)
        assert len(ids) == 2, f"The internal point is not in 2 contours, but {len(ids)}"
        point_a, point_b = find_closest_points_between_two_contours(
            obs_map, contours_list[ids[0]], contours_list[ids[1]]
        )
        obs_map = cv2.line((obs_map * 255).astype(np.uint8), (point_a[1], point_a[0]), (point_b[1], point_b[0]), 255, 5)
        obs_map = obs_map == 255
        contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
           obs_map, 0, detect_internal_contours=False
        )

    poly_list = []

    for contour in contours_list:
        #if is_near_crop_boundary(contour, 0, 254, 0, 167, margin=2):
        #    continue
         
        if vis:
            contour_cv2 = contour[:, [1, 0]]
            cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)
            cv2.imshow("obs_second", obs_map_vis)

        contour_pos = []
        for [row, col] in contour:
            contour_pos.append(vg.Point(row, col))
        poly_list.append(contour_pos)

        xlist = [x.x for x in contour_pos]
        zlist = [x.y for x in contour_pos]
        if vis:
             plt.plot(xlist, zlist)
             cv2.waitKey(0)
             #cv2.destroyWindow("obs")

    g = vg.VisGraph()
    g.build(poly_list, workers=4)
    return g  

def get_nearby_position(goal: Tuple[float, float], G: vg.VisGraph,
                        obstacles:np.ndarray, patch_size: int = 3,
                        max_radius: int = 20) -> Tuple[float, float]:
    #main.py 수정

    H, W = obstacles.shape
    offset = patch_size // 2
    visited = np.zeros_like(obstacles, dtype=bool)

    start_r, start_c = int(goal[0]), int(goal[1])
    queue = deque()
    queue.append((start_r, start_c, 0))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    #print("run")
    while queue:
        r, c, radius = queue.popleft()

        if not (0 <= r < H and 0 <= c < W):
            #print(f"[SKIP] Out of bounds: ({r}, {c})")
            continue
        if visited[r, c]:
            #print(f"[SKIP] Already visited: ({r}, {c})")
            continue
        visited[r, c] = True

        goalvg_new = vg.Point(r, c)
        poly_id = G.point_in_polygon(goalvg_new)
        if poly_id == -1:
            rmin = max(0, r - offset)
            rmax = min(H, r + offset + 1)
            cmin = max(0, c - offset)
            cmax = min(W, c + offset + 1)
            
            patch = obstacles[rmin:rmax, cmin:cmax]
            #print(f"[CHECK] ({r}, {c}), radius: {radius}, patch size: {patch.shape}, free: {np.all(patch == 1)}")

            if np.all(patch == 1):
                return (r,c)
            
        if radius < max_radius:
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                queue.append((nr, nc, radius + 1))
        
    #print(f"[WARNING] No valid nearby position found near goal: {goal}")
    return goal

def plan_to_pos_v2(start, goal, obstacles, G: vg.VisGraph = None, vis=False):
    """
    plan a path on a cropped obstacles map represented by a graph.
    Start and goal are tuples of (row, col) in the map.
    """

    print("start: ", start)
    print("goal: ", goal)
    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 3, (255, 0, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 3, (0, 0, 255), -1)
        cv2.imshow("planned path", obs_map_vis)
        cv2.waitKey()

    path = []
    startvg = vg.Point(start[0], start[1])
    if obstacles[int(start[0]), int(start[1])] == 0:
        print("start in obstacles")
        rows, cols = np.where(obstacles == 1)
        dist_sq = (rows - start[0]) ** 2 + (cols - start[1]) ** 2
        id = np.argmin(dist_sq)
        new_start = [rows[id], cols[id]]
        path.append(new_start)
        startvg = vg.Point(new_start[0], new_start[1])

    goalvg = vg.Point(goal[0], goal[1])
    poly_id = G.point_in_polygon(goalvg)
    if obstacles[int(goal[0]), int(goal[1])] == 0:
        print("goal in obstacles")
        try:
            goalvg = G.closest_point(goalvg, poly_id, length=1)

            #main.py 추가 - goalvg 좌표가 유효한지 확인
            if goalvg.x <= 0 or goalvg.y <= 0:
                print(f"[WARNING] closest_point returned invalid goalvg: ({goalvg.x:.2f}, {goalvg.y:.2f})")
                goal_new = get_nearby_position(goal, G , obstacles, patch_size=3)
                goalvg = vg.Point(goal_new[0], goal_new[1])
        except:
            goal_new = get_nearby_position(goal, G , obstacles, patch_size=3)
            goalvg = vg.Point(goal_new[0], goal_new[1])


        print("origin_goalvg: ", goalvg)

        if goalvg.y <= 75:
            if 90 <= goalvg.x <= 120 and goalvg.y > 60: # stove
                print(f"[INFO] Adjusting goalvg.x to {goalvg.x:.2f},goalvg.y to {goalvg.y-20:.2f}")
                goalvg = vg.Point(goalvg.x, goalvg.y-20)
            elif 120 <= goalvg.x <= 140:
                if goalvg.y < 45: #sink in kitchen
                    print(f"[INFO] Adjusting goalvg.x to {goalvg.x-30:.2f},goalvg.y to {goalvg.y:.2f}")
                    goalvg = vg.Point(goalvg.x-30, goalvg.y)
                elif 60 > goalvg.y >= 45: # sink in kitchen
                    print(f"[INFO] Adjusting goalvg.x to {goalvg.x-20:.2f},goalvg.y to {goalvg.y:.2f}")
                    goalvg = vg.Point(goalvg.x-20, goalvg.y)
                elif goalvg.y >= 60: # stove
                    print(f"[INFO] Adjusting goalvg.x to {goalvg.x-10:.2f},goalvg.y to {goalvg.y-18:.2f}")
                    goalvg = vg.Point(goalvg.x-10, goalvg.y-18)
            elif 140 <= goalvg.x <= 160: # sink in bathroom & toliet
                print(f"[INFO] Adjusting goalvg.x to {goalvg.x+20:.2f},goalvg.y to {goalvg.y:.2f}")
                goalvg = vg.Point(goalvg.x+20, goalvg.y)
            elif 180 <= goalvg.x <= 190: # shelving in study room
                print(f"[INFO] Adjusting goalvg.x to {goalvg.x+20:.2f},goalvg.y to {goalvg.y+10:.2f}")
                goalvg = vg.Point(goalvg.x+30, goalvg.y+10)
        elif goalvg.y > 160:
            if 125 <goalvg.x < 140:
                print(f"[INFO] Adjusting goalvg.x to {goalvg.x-15:.2f},goalvg.y to {goalvg.y-20:.2f}")
                goalvg = vg.Point(goalvg.x-17, goalvg.y-20)
            else:
                print(f"[INFO] Adjusting goalvg.x to {goalvg.x:.2f},goalvg.y to {goalvg.y-20:.2f}")
                goalvg = vg.Point(goalvg.x, goalvg.y-20)

        if goalvg.x > 250:
            print(f"[INFO] Adjusting goalvg.x to {goalvg.x-20:.2f},goalvg.y to {goalvg.y:.2f}")
            goalvg = vg.Point(goalvg.x-20, goalvg.y)

        
    path_vg = G.shortest_path(startvg, goalvg)

    for point in path_vg:
        subgoal = [point.x, point.y]
        path.append(subgoal)
    print(path)

    # check the final goal is not in obstacles
    # if obstacles[int(goal[0]), int(goal[1])] == 0:
    #     path = path[:-1]

    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])

        for i, point in enumerate(path):
            subgoal = (int(point[1]), int(point[0]))
            print(i, subgoal)
            obs_map_vis = cv2.circle(obs_map_vis, subgoal, 5, (255, 0, 0), -1)
            if i > 0:
                cv2.line(obs_map_vis, last_subgoal, subgoal, (255, 0, 0), 2)
            last_subgoal = subgoal
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 5, (0, 255, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 5, (0, 0, 255), -1)

        seg = Image.fromarray(obs_map_vis)
        cv2.imshow("planned path", obs_map_vis)
        cv2.waitKey()

    return path


def get_bbox(center, size):
    """
    Return min corner and max corner coordinate
    """
    min_corner = center - size / 2
    max_corner = center + size / 2
    return min_corner, max_corner


def get_dist_to_bbox_2d(center, size, pos):
    min_corner_2d, max_corner_2d = get_bbox(center, size)

    dx = pos[0] - center[0]
    dy = pos[1] - center[1]

    if pos[0] < min_corner_2d[0] or pos[0] > max_corner_2d[0]:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
            *  |  |  *
            ___|__|___
               |  |
            ___|__|___
               |  |
            *  |  |  *
            """

            dx_c = np.abs(dx) - size[0] / 2
            dy_c = np.abs(dy) - size[1] / 2
            dist = np.sqrt(dx_c * dx_c + dy_c * dy_c)
            return dist
        else:
            """
            star region
               |  |
            ___|__|___
            *  |  |  *
            ___|__|___
               |  |
               |  |
            """
            dx_b = np.abs(dx) - size[0] / 2
            return dx_b
    else:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
               |* |
            ___|__|___
               |  |
            ___|__|___
               |* |
               |  |
            """
            dy_b = np.abs(dy) - size[1] / 2
            return dy_b

        """
        star region
           |  |  
        ___|__|___
           |* |   
        ___|__|___
           |  |   
           |  |  
        """
        return 0
