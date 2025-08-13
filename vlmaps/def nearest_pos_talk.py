map.py / def get_nearest_po

def get_nearest_pos(self, curr_pos: List[float], name: str) -> List[float]:
        contours, centers, bbox_list = self.get_pos(name)
        ids_list = self.filter_small_objects(bbox_list, area_thres=10)

        # 수정사항 1 : 객체들의 중심점을 출력해보기
        print(f"[Filtered Centers for {name}]")
        for idx, center in enumerate(centers):
            print(f"  Object {idx}: center = {center}")
        ##################################
        
        contours = [contours[i] for i in ids_list]
        centers = [centers[i] for i in ids_list]
        bbox_list = [bbox_list[i] for i in ids_list]
        if len(centers) == 0:
            return curr_pos
        

        #수정사항 3 : 정사각형 바운더리 설정 후, 바운더리 내의 객체로 이동하기
        #사용자가 바운더리 중심의 좌표를 입력하면, 중심을 기준으로 상하좌우 20셀(40*40 크기) 바운더리 설정
        #user_center_input = input("Enter boundary center position as 'row,col' (default=skip): ").strip()
        if user_center_input != "":
            try:
                row_str, col_str = user_center_input.split(",")
                b_row = float(row_str)
                b_col = float(col_str)
                boundary_half_width = 20  # 셀 단위
                row_min = b_row - boundary_half_width
                row_max = b_row + boundary_half_width
                col_min = b_col - boundary_half_width
                col_max = b_col + boundary_half_width

                in_boundary_ids = []
                for i, center in enumerate(centers):
                    row, col = center
                    if row_min <= row <= row_max and col_min <= col <= col_max:
                        in_boundary_ids.append(i)

                if not in_boundary_ids:
                    print(f"[WARN] No objects found within boundary centered at ({b_row}, {b_col})")
                    return curr_pos

                # 바운더리 내 객체만 필터링
                contours = [contours[i] for i in in_boundary_ids]
                centers = [centers[i] for i in in_boundary_ids]
                bbox_list = [bbox_list[i] for i in in_boundary_ids]

            except Exception as e:
                print(f"[ERROR] Failed to parse input. Ignoring boundary filter. ({e})")

        print(f"[Boundering Centers for {name}]")
        for idx, center in enumerate(centers):
            print(f"  Object {idx}: center = {center}")
        ##################################


        # 수정사항 2 : 가장 가까운 객체가 아닌, 사용자가 지정한 객체로 이동하기
        selected_id = None
        user_input = input(f"\nEnter object index to move to (default=nearest): ").strip()

        if user_input != "":
            try:
                parsed = int(user_input)
                if 0 <= parsed < len(centers):
                    selected_id = parsed
                else:
                    print(f"Out of range. Falling back to nearest object.")
            except ValueError:
                print(f"Invalid input. Falling back to nearest object.")


        if selected_id is not None:
            print(f"[INFO] Using user-selected object ID: {selected_id}")
            id = selected_id
        else:
            id = self.select_nearest_obj(centers, bbox_list, curr_pos)
            print(f"[INFO] No valid selected_id provided. Nearest object selected: {id}")
        ##################################
        
        return self.nearest_point_on_polygon(curr_pos, contours[id])
