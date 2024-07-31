## 数据采集流程
**NOTE** 以下代码每次运行都会覆盖原本文件，因此注意路径的设置
1. 运行 projects/habitat_ovmm/gather_episode.py 过滤掉重复的episode
1. 运行 projects/habitat_ovmm/place_data_collection.py 收集容器位置,
```
receptacle_position_aggregate(args.data_dir, env)
```
需要被解注释
2. 运行 projects/habitat_ovmm/place_data_collection.py 收集放置容器位置（其保留所有交互成功的数据，按照一定概率保留交互失败的数据）,脚本如下：
```
python projects/habitat_ovmm/place_data_collection.py --data_dir cyw/datasets/place_dataset/train --baseline_config_path cyw/configs/agent/rl_agent_place.yaml --env_config_path cyw/configs/env/hssd_eval_gtseg.yaml habitat.dataset.split=train +habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map +habitat.task.measurements.top_down_map.meters_per_pixel=0.05
```
3. 运行 cyw/goal_point/data_prepare.py 将收集到的数据进行处理，转为局部地图，以及target map
4. 运行 gen_training_data_v1/assignment_id.py 给每条数据分配一个id

**现在（2024-07-01）数据的缺点**：
1. 如果用rl采集数据，由于rl会走动，导致采集到的waypoint不再是围绕recep的点，start rgb不包含这些信息，对模型来说，就是瞎猜
2. 采集速度非常慢，一个晚上只能采2个episode的数据

**NOTE**
代码中设置 `collect_fail_prob = 0.1 # 当失败时，以0.1的概率采集数据`, 最好先设置为1，测试一下 sr（cyw/goal_point/count_sr.py），然后再设置一个比较好的比例，让数据类别均衡

**如何才能使机器人选一个好交互的容器进行交互**

## 数据说明
1. 数据组织结构
```
root_dir(cyw/datasets/place_dataset):
    split(train,val):
        recep_position.pickle
        agent(rl_agent_place,heuristic_...,...)
            data_out.hdf5
            place_waypoint.pkl
            prepared_dataset.hdf5
```
2. 各个数据说明
2.1 data_out.hdf5 为 projects/habitat_ovmm/place_data_collection.py 生成的文件，包含：
```
    scene_ep_recep_grp.create_dataset(name="start_rgb_s",data=start_rgb_s)
    scene_ep_recep_grp.create_dataset(name="start_semantic_s",data=start_semantic_s)
    scene_ep_recep_grp.create_dataset(name="start_depth_s",data=start_depth_s)
    scene_ep_recep_grp.create_dataset(name="start_top_down_map_s",data=start_top_down_map_s)
    scene_ep_recep_grp.create_dataset(name="start_obstacle_map_s",data=start_obstacle_map_s)
    scene_ep_recep_grp.create_dataset(name="view_point_position_s",data=view_point_position_s)
```
view_point_position_s 是世界坐标系下的位置，此处用作各条数据的id，用来验证数据对齐准确性
2.2 place_waypoint.pkl记录了各个viewpoint在世界坐标系的位置，以及recep相对当前viewpoint的位置（用于验证后面坐标转换的正确性）
```
    skill_waypoint_singile_recep_data["each_view_point_data"].append(
        {
            "view_point_position":view_point_position,
            "start_position":start_position,
            "start_rotation":start_rotation,
            "relative_recep_position": relative_recep_gps,
            "end_position": end_position,
            "place_success": place_success,
            "start_sensor_pose": start_sensor_pose, # 在obstacle map里面的位置
            "start_top_down_map_pose": start_top_down_map_pose,
            "start_top_down_map_rot": start_top_down_map_rot
        }
    )
```
start_position，start_rotation 世界坐标系下机器人的位置，rotation，三维空间，用于后续将其它位置坐标转为相对坐标
end_position 结束时，机器人在世界坐标系下的位置，后续会通过utils.get_relative_position 计算相对位置
start_sensor_pose： sem_map返回的[x,y,0,local_map_boundary]，与gps同一坐标系，在sem_map中的坐标还需要转换
start_top_down_map_pose，start_top_down_map_rot: top_down_map的机器人位置，朝向
2.3 prepared_dataset.hdf5 为经过预处理后的数据
```
scene_ep_recep_grp_processed.create_dataset(name="local_top_down_map_s",data=local_top_down_map_s)
scene_ep_recep_grp_processed.create_dataset(name="local_obstacle_map_s",data=local_obstacle_map_s)
scene_ep_recep_grp_processed.create_dataset(name="target_s",data=target_s)
scene_ep_recep_grp_processed.create_dataset(name="recep_coord_s",data=recep_coord_s)
scene_ep_recep_grp_processed.create_dataset(name="waypoint_s",data=waypoint_s)
```
local_top_down_map_s 为将 环境返回 的top_down_map 进行旋转，使得机器人朝向为 ➡，为使图像与obstacle对应，这里进行了flipup, 然后截取机器人附近一个小片段的图像
local_obstacle_map_s 为 将 sem_map返回的 obstacle_map 进行旋转，使得机器人朝向为 ➡ （注意，此时图像与gps坐标系存在上下翻转的关系）， 然后截取机器人附近一个小片段的图像
target_s 交互成功的点，方位关系与 local_obstacle_map_s 一致
recep_coord_s： 容器在图中的坐标 (row_index,col_index)
waypoint_s: view_point的世界坐标系下的坐标，验证数据准确性使用

## 坐标系总结
1. top_down_map 模块
⬇ x  ➡ y  theta 为与 x轴正方向的夹角，逆时针为正，顺时针为负
2. sem_map模块
sem_map 建图模块 的坐标系应该是和 gps坐标系一致，因为其 pose_delta是由gps坐标变化计算而来
gps 坐标系：➡ x  ⬆ y
但是需要注意的是，sem_map计算得到的图是真实障碍物的上下翻转版本，因此所有可视化都需要对地图上下翻转，但是机器人的位置还是按照 gps坐标系来画（只是因为cv2坐标系为 ➡x ⬇ y , 所以作图的时候对agent pose进行了坐标变换）
3. fmm_planner坐标系
使用索引作为坐标，即：⬇ x ➡ y ,至于为什么 start 只是简单的 把 sensor pose换算为网格的（x,y）对调？ 因为sem_map是上下翻转的
fmm planner坐标系对应索引坐标系，如下:
```
    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
    start_o 为 与 x轴正轴的夹角
    ...
    start = [
    int(start_y * 100.0 / self.map_resolution - gx1),
    int(start_x * 100.0 / self.map_resolution - gy1),
    ]
```
但是，由于夹角需要与gps坐标系下的夹角对应，因此，夹角的计算看起来有些不同寻常
```
angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
relative_angle_to_stg = pu.normalize_angle(angle_agent - angle_st_goal)
```
此时的夹角是与 ➡ 正方向的夹角 逆时针为正，但是，由于整个地图是现实世界的翻转版本，因此，在地图上算出的正夹角，在现实执行动作时，应该旋转负夹角
```
if relative_angle_to_stg > self.turn_angle / 2.0:
    action = DiscreteNavigationAction.TURN_RIGHT
```
**综上所述** 以 obstacle_map作为输入，预测这个地图上的点作为goal点（此时的goal点与gps goal点存在上下翻转的关系），直接调用 fmm_planner规划路径即可；如果使用top_down_map作为输入，其给出的goal点，还需要进行上下翻转才能给fmm进行路径规划

## NOTE
place waypoint文件用于分配id，不能删除