# 数据采集详细说明
## 采集流程
```
for episode:
    recep = get_recep(episode)
    scene = get_scene(episode)
    recep_pos_s = get_recep_pos(scene,recep)
    for recep_pos:
        view_points = get_viewpoints(recep_pos)
        for view_point:
            initial_pos(view_point)
            pickup_obj()
            while not done:
                action = agent(obs)
                env.step(action)
            success = get_place_success()
            record_data() # rgb semantic depth map agent_start_world_pos agent_end_world_pos recep_world_pos
            agent.reset()
            env.reset_state() # 重置评价指标,但不加载下一条轨迹
```
**如何将每条轨迹尝试的waypoint转换在同一坐标系下?**

以agent当前位置,朝向建立坐标系(前向为x轴,左边为y轴,与gps坐标设置一致)
```
relative_position = quaternion_rotate_vector(
    rotation_world_current.inverse(), position - origin
    )
```
仿真环境返回的坐标是三维坐标,使用四元数进行坐标变换
走的弯路: 试图利用gps,compass进行坐标转换,但最终发现gps,compass是相对agent初始位置的坐标,无法完成
![Alt text](image/waypoint_image.png)

### 采集地图
1. top_down_map

![Alt text](image/top_down_map_initial.jpg)

![Alt text](image/top_down_coord.png)
(x,y,rot)

旋转后的地图
![Alt text](image/0/top_down_map_vis.jpg)

交互成功的viewpoint

![Alt text](image/0/top_down_map_waypoint.jpg)

对比 rgb
![Alt text](image/0/rgb_vis.jpg)

2. semmapping建立的 obstacle map

![Alt text](image/0/init_obstacle_map_vis.jpg)

(x,y,rot)

**坐标在图中的什么位置?**
查看建图模块,其坐标通过gps差值计算而来,因此其应该和gps使用同一坐标系
![Alt text](image/obstacle_image.png)
坐标应该是(h-y,x)

fmm planner计算坐标:
```
    start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
    start_o 为 与 x轴正轴的夹角
    ...
    start = [
    int(start_y * 100.0 / self.map_resolution - gx1),
    int(start_x * 100.0 / self.map_resolution - gy1),
    ]
```
经对比可视化时将所有图片flipup,以及路径规划中
```
angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
relative_angle_to_stg = pu.normalize_angle(angle_agent - angle_st_goal)
```
```
if relative_angle_to_stg > self.turn_angle / 2.0:
    action = DiscreteNavigationAction.TURN_RIGHT
```
推测 obstacle和 gps(以及与现实) 左右方位相反, 在obstacle map中,左边为agent朝向顺时针方向

![Alt text](image/obstacle_direction.png)

obstacle map 与 top_down_map 翻转后稍微能对应

![Alt text](image/0/loacal_rom_vis.jpg)
![Alt text](image/0/local_rtdm_vis.jpg)
![Alt text](image/0/local_rom_recep_vis.jpg)

## 数据量大,数据采集慢
一个episode, 有十几个recep instance, 每个recep instance 有 成百个 viewpoint, 每个viewpoint 300-400 步, 结果,采一个episode的数据大概需要 3-4 小时, 数据量大概有 6 G, 然而,train split有 30000+ episode, val 有 2000左右 episode

目前已经将episode过滤,去除掉在同一scene,操作同一recep(认为要放的东西不重要)的episode, 剩余200+ episodes, 然而仍然需要很久才能采集完

所以,考虑每个recep instance随机采样一部分viewpoint?
