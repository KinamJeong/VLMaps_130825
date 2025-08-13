def build_visgraph_with_obs_map(obs_map, use_internal_contour=False, internal_point=None, vis=False):
    obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
    obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
    #1. 기본 장애물 맵 시각화
    if vis:
       cv2.imshow("obs", obs_map_vis)
       cv2.waitKey()
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

    #3. 장애물 contour 시각화
    for contour in contours_list:
        if vis:
            contour_cv2 = contour[:, [1, 0]]
            cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)
            cv2.imshow("obs", obs_map_vis)
        contour_pos = []
        for [row, col] in contour:
            contour_pos.append(vg.Point(row, col))
        poly_list.append(contour_pos)
        xlist = [x.x for x in contour_pos]
        zlist = [x.y for x in contour_pos]
        if vis:
             plt.plot(xlist, zlist)
             cv2.waitKey()
             #cv2.destroyWindow("obs")
    g = vg.VisGraph()
    g.build(poly_list, workers=4)
    return g
