## requirement
按照 https://ollama.com/blog/llama3 部署 llama3
参考 https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes 部署yolo系列 环境
注意：yolo-world本身环境部署报错

## update
- 2024-04-20 /raid/cyw/home-robot/home-robot/src/third_party/detectron2/detectron2 在这个文件夹下面git clone了(detectron2)[https://detectron2.readthedocs.io/en/latest/tutorials/install.html]项目；/raid/home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/third_party/ 下载了(CenterNet2)[https://github.com/xingyizhou/CenterNet2/tree/master]
- 2024-04-24 
1. 增加src/home_robot/home_robot/navigation_policy/object_navigation/llava_exploration_policy.py 想用llava来根据当前图像选择下一个探索点，这样没办法解决历史记忆问题，先放弃

- 2024-04-25 
1. 增加cyw/configs/agent/heuristic_agent_esc.yaml，将semantic_map 的num_sem_categories改为22（应该改为24，真实种类+2），修改navobj navrec的类别
 src/home_robot/home_robot/agent/ovmm_agent/ovmm_agent.py 文件里面增加esc相关设置

 - 2024-04-26 
 1. 修改了配置文件record_instance_ids（已改回来，这部分代码似乎有点问题，调用不了）
 2. 修改esc_agent里面 store_all_categories: False  # whether to store all semantic categories in the map or just task-relevant ones 为True(已修改回来，设置为True之后，会检测所有物体)
 3. 分别在ovmm agent, objnav agent, objnav module, visualize里面加上debug（已都置为false，如需调试，可设置为true）
 4. 修改object nav agent _preprocess_obs，增加了一个属性vocab_full，将所有容器的senmantic传到语义建图模块

 - 2024-04-28
 1. 安装ollama, 基于其部署了llama3，参考https://github.com/ollama/ollama?tab=readme-ov-file
    写prompt /raid/home-robot/cyw/llama3_utils/prompt.py 获得llama3-8b响应，使用 cyw/llama3_utils/parse_response.py解析相应，结果文件存储在cyw/data 中
 2. 修改detection的vocab及配置文件，增加对房间的识别及建图，但detic模型基本认不出房间

 - 2024-04-30
 1. psl单独调通
 2. 备份 src/home_robot/home_robot/navigation_policy/object_navigation/objectnav_ESC_policy.py 文件

 - 2024-05-04
 1. 增加psl近似计算代码，其计算结果似乎和严格计算没区别，各方面调试通

 - 2024-05-06
 1. 备份 src/home_robot/home_robot/agent/objectnav_agent/objectnav_agent_20240506.py

 - 2024-05-07
 1. psl 迁移完成，能够跑起来，但是运行起来有些奇怪，还需调试

 - 2024-05-22
 1. 将place 任务单独拎出来，主要是修改配置文件，其中因为 place 任务 动作空间和观测空间都与 ovmm 任务不同，因此，创建了 一个 place_agent 和 place_env， 文件位置与 ovmm项目相关文件并列。配置文件的修改具体如下：
 sensor修改：尽可能地把所有能获取的观测都加进去
 act 修改： max_forward, max_turn, max_joint_delta，因为动作转换的时候会用到，因此 将它们修改为与ovmm 一致。但是目前动作空间似乎还是有些问题，输出 stop 的时候，环境没有stop
 另：如果要看到 third_rgb ，不是在配置文件里修改，将代码运行参数 env_eval 设置为hssd_demo 即可（虽然不知道为啥会这样）
  
 
## TODO
1. src/home_robot/home_robot/navigation_policy/object_navigation/objectnav_frontier_exploration_policy.py 计算前端点没有考虑障碍物，只考虑了探索过的区域的边缘，是否改为和ESC一样，去掉障碍物
2. 到达目标点后再请求下一个目标点？
3. PSL探索策略里面，筛去距离进的，是否靠近物体，是否靠近房间，计算太过于绝对，有很多超参数，是否有改进方法？
4. record_instance_id代码没看懂，src/home_robot/home_robot/navigation_policy/object_navigation/objectnav_frontier_exploration_policy.py 代码中对instance_id的处理跳过了
5. cyw/configs/agent/heuristic_agent_esc.yaml 里 语义图是否记录房间单独设置了，但是这些参数应该是级联变化
6. 现在只要是esc 探索，都建立完整的语义图，不太合理，因为如果reasoning是room的话，没有必要建立其它物体的语义图
```
        self.full_vocab = getattr(config.AGENT.SKILLS.NAV_TO_OBJ,"type","heuristic")=="heuristic_esc" #在语义地图里面记录所有的大物体
```

## NOTE
1. 如增加了房间识别，可视化也要相应的修改
2. habitat env 配置嵌套太严重，直接在相应目录下修改，主要包括：src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/ovmm/ovmm.yaml 以及 src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/ovmm/ovmm.yaml


 ## info 
 1. visualize文件夹 src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/visualizer.py

 2. src/home_robot/home_robot/perception/constants.py 记录了很多种物体类别

 3. semmap 613行：voxels = du.splat_feat_nd(init_grid, feat, XYZ_cm_std).transpose(2, 3)
 把初始化的网格，语义特征，点云图结合起来，形成了体素图

 4. 
 ```
# Local obstacles, explored area, and current and past position
map_features[:, 0 : MC.NON_SEM_CHANNELS, :, :] = local_map[
   :, 0 : MC.NON_SEM_CHANNELS, :, :
]
# Global obstacles, explored area, and current and past position
map_features[
   :, MC.NON_SEM_CHANNELS : 2 * MC.NON_SEM_CHANNELS, :, :
] = nn.MaxPool2d(self.global_downscaling)(
   global_map[:, 0 : MC.NON_SEM_CHANNELS, :, :]
)
# Local semantic categories
map_features[:, 2 * MC.NON_SEM_CHANNELS :, :, :] = local_map[
   :, MC.NON_SEM_CHANNELS :, :, :
]
```
map_features 前6个通道是local map, 后6个通道是global map。global_map 的loacal map boundary为local map, 似乎并不是每一步都更新global map, global map也没有全局的语义，有什么用吗？

5. 建图和选择goal policy都可以多物体，但是plan只能一个
```
(
      action,
      closest_goal_map,
      short_term_goal,
      dilated_obstacle_map,
) = self.planner.plan(
      **planner_inputs[0],
      use_dilation_for_stg=self.use_dilation_for_stg,
      timestep=self.timesteps[0],
      debug=self.verbose,
)
```

6. 语义图module给出的 seq_map_features, local_map, global_map都是(num_catogery,h,w)，包含了多加的两类other, misc, 传到后面模块的图也是这样的，因此需要格外注意objcet_idx

7. src/home_robot/home_robot/navigation_planner/fmm_planner.py 里有对fmm_dist的可视化，可模仿学习

8. fmm_planner 和 visualize 都把图像反转过来了 np.flipud(semantic_map_vis), 不知道为啥

9. plt 可视化时，plt.show()是一个阻塞函数，只有当窗口被关闭后才能运行后面的内容，同时显示的图像也不会随着程序的运行更新（不像cv2）, 只有当程序暂停的时候才会更新

10. 似乎确实没有多线程版本，readme里面也没说，真要用多线程跑的话，结果文件的记录，都需要补充

11. projects/habitat_ovmm/evaluator.py 有写一个智能体只能工作一个环境

12. src/third_party/habitat-lab/habitat-baselines/habitat_baselines/common/habitat_env_factory.py 环境初始化函数 vector envrioment

13. 传入的env 配置经过 projects/habitat_ovmm/utils/config_utils.py 包装配置, 可以在里面设置级联参数,目前不是很有必要

14. habitat_config文件 src/third_party/habitat-lab/habitat-baselines/habitat_baselines/config/ovmm/ovmm_eval.yaml

15. 现在的代码原则上可以用一个agent处理多个环境，但是目前agent的识别模块设置了与任务相关的识别，当另一个环境给出新的任务时，agent报错

16. dataset配置主要在habitat_config文件里面，然后habitat_config前几行调用了 default配置参数，引用了其它文件，相关文件基本位于： src/third_party/habitat-lab/habitat-lab/habitat/config/benchmark/ovmm/ovmm.yaml  可用linux系统 find . -d type -nmae *** 查找相关配置文件

17. 默认配置里面跳过了gaze
```
  gaze_at_obj: True
  gaze_at_rec: True
```
这个步骤

18. 走的过程会把close区域移除：src/home_robot/home_robot/navigation_policy/object_navigation/objectnav_frontier_exploration_policy.py 278行

19. agent配置里面默认设置   use_dilation_for_stg: False 这时descrite planner 会选择一个最近的可导航的目标点，但是这似乎没有起到效果？ 为什么会出现卡死，可能需要更详细地找找原因

20. 默认放置设置：启发式放置

21. 应该好好看看安装路径 /raid/home-robot/install_deps.sh

22. baseline rl nav使用离散动作，place and pick 使用连续动作

23. src/home_robot/home_robot/manipulation/heuristic_pick_policy.py 有单独的启发式抓和放策略

24. src/third_party/habitat-lab/test/test_habitat_task.py 这里面有用连续动作的代码

25. src/third_party/habitat-lab/habitat-lab/habitat/config/CONFIG_KEYS.md 有关于orcal navigation的说明（或许应该看看RL训练pick，place之类的东西）

26. src/third_party/habitat-lab/habitat-baselines/habitat_baselines/run.py 是ppo训练代码

27. OVMMFindRecepPhaseSuccess 指的是 机器人拿起东西，导航到容器success distance以内，并且在 success_angle 以内朝向容器

28. gym 强化学习环境和一般的环境好像不同

29. projects/habitat_ovmm/receptacles_data_collection.py 有搜集容器数据的代码

30. projects/habitat_objectnav/files_to_adapt/eval_env_wrapper.py 计算了 episode, 里面有场景、地图

31. habitat-lab/habitat/config/CONFIG_KEYS.md 配置文件说明要好好读一读，比如有如下关键信息： |habitat.task.actions.oracle_nav_action| Rearrangement Only, Oracle navigation action. This action takes as input a discrete ID which refers to an object in the PDDL domain. The oracle navigation controller then computes the actions to navigate to that desired object.| 

32. ovmm_agent注释里面有写： # currently we get ground truth semantics of only the target object category and all scene receptacles from the simulator 只能得到目标物体类别和容器的语义

33. 判断是否成功的指标在： projects/habitat_ovmm/utils/metrics_utils.py 中

34. gt semantic中 0对应背景，1对应小物体，2：对应 大物体

35. docs/challenge.md 有提交的详细说明

36. vocabu除了让模型检测外，还有什么用？ 在src/home_robot/home_robot/perception/wrapper.py中，将 goal start_recp和end_recp 根据名称计算id。后续导航模型会根据记录的物体类别数再进行调整

37. 在技能之间转换时，也会 reset vocab

38. 旋转角度 30 度, 前进一步是0.25

39. projects/real_world_ovmm/configs/example_cat_map.json 记录了所有的obj 与id的对应关系

40. src/third_party/habitat-lab/habitat-lab/habitat/tasks/rearrange/sub_tasks/nav_to_obj_sensors.py 这个文件定义了 robot_start_gps 的计算方式

41. start_gps (也是ovmm返回的gps) 计算方式：
   ```
          def get_observation(
        self, observations, episode, task, *args: Any, **kwargs: Any
    ):
        start_position = self.get_agent_start_position(episode, task)
        rotation_world_start = self.get_agent_start_rotation(episode, task)

        origin = np.array(start_position, dtype=np.float32)

        agent_position = self.get_agent_current_position(self._sim)

        relative_agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array(
                [-relative_agent_position[2], relative_agent_position[0]],
                dtype=np.float32,
            )
        else:
            return relative_agent_position.astype(np.float32)
   ```
   按照起始位置，起始朝向建立坐标系， robot_start_gps 是在这一坐标系下的坐标位置（这段代码在 src/third_party/habitat-lab/habitat-lab/habitat/tasks/nav/nav.py 中）

42. compass 计算方式
   ```
    def get_observation(
            self, observations, episode, task, *args: Any, **kwargs: Any
        ):
            rotation_world_start = self.get_agent_start_rotation(episode, task)
            rotation_world_agent = self.get_agent_current_rotation(self._sim)

            if isinstance(rotation_world_agent, quaternion.quaternion):
                return self._quat_to_xy_heading(
                    rotation_world_agent.inverse() * rotation_world_start
                )
            else:
                raise ValueError("Agent's rotation was not a quaternion")
   ```
   似乎相当于 在起始坐标系下的 朝向？ src/third_party/habitat-lab/habitat-lab/habitat/tasks/nav/nav.py 
   具体 agent pos 和 agent rot 计算方式如下

   ```
    @registry.register_sensor(name="RobotStartGPSSensor")
    class RobotStartGPSSensor(EpisodicGPSSensor):
        cls_uuid: str = "robot_start_gps"

        def __init__(self, sim, config: "DictConfig", *args, **kwargs):
            super().__init__(sim=sim, config=config)

        def get_agent_start_position(self, episode, task):
            return task._robot_start_position

        def get_agent_start_rotation(self, episode, task):
            return quaternion_from_coeff(task._robot_start_rotation)

        def get_agent_current_position(self, sim):
            return sim.articulated_agent.sim_obj.translation
   ```
    and
   ```
        @registry.register_sensor(name="RobotStartCompassSensor")
    class RobotStartCompassSensor(EpisodicCompassSensor):
        cls_uuid: str = "robot_start_compass"

        def __init__(self, sim, config: "DictConfig", *args, **kwargs):
            super().__init__(sim=sim, config=config)

        def get_agent_start_rotation(self, episode, task):
            return quaternion_from_coeff(task._robot_start_rotation)

        def get_agent_current_rotation(self, sim):
            curr_quat = sim.articulated_agent.sim_obj.rotation
            curr_rotation = [
                curr_quat.vector.x,
                curr_quat.vector.y,
                curr_quat.vector.z,
                curr_quat.scalar,
            ]
            return quaternion_from_coeff(curr_rotation)
   ```
   相关代码在 src/third_party/habitat-lab/habitat-lab/habitat/tasks/rearrange/sub_tasks/nav_to_obj_sensors.py 中

43. src/third_party/habitat-lab/habitat-lab/habitat/tasks/rearrange/sub_tasks/nav_to_obj_task.py 中相关初始化代码，似乎有机器人初始位置放置，以及物体摆放

44. sim.articulated_agent.sim_obj.translation 似乎和 sim.articulated_agent.base_pose 一样

45.   hfov: 42.0              # horizontal field of view (in degrees) env里面指明了水平视场角 为 42 度

46. top_down_map的计算方式在 src/third_party/habitat-lab/habitat-lab/habitat/tasks/nav/nav.py 中

47. semantic map里面，原点都是 0，但是把开始位置初始化为地图中点
```
    p = map_size_parameters
    global_pose[e].fill_(0.0)
    global_pose[e, :2] = p.global_map_size_cm / 100.0 / 2.0

    # Initialize starting agent locations
    x, y = (global_pose[e, :2] * 100 / p.resolution).int()
    global_map[e].fill_(0.0)
    global_map[e, 2:4, y - 1 : y + 2, x - 1 : x + 2] = 1.0
```
src/home_robot/home_robot/mapping/map_utils.py

48. cv2 里面 坐标轴： ➡ x
                    ⬇ y

49. 一开始的时候先向右旋转，右 为顺时针 左 为逆时针

50. recenter_local_map_and_pose_for_env 会移动 obstacle map的中心  src/home_robot/home_robot/mapping/semantic/categorical_2d_semantic_map_module.py

51. nac_to_place 怎么计算：
    距离和角度都在一定范围内，其中goal点设置为 viewpoint
        return np.stack(
            [
                view_point.agent_state.position
                for goal in episode.candidate_goal_receps
                for view_point in goal.view_points
            ],
            axis=0,
        )
    朝向是计算到容器的中心的
    """Angle between agent's forward vector and the vector from agent's position to the center of the closest candidate place receptacle"""
    所以即使按照环境的viewpoint初始化，仍然有些情况没有朝向容器


## 各个模块坐标系整理
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
**经验证，以上坐标系猜想正确**

如果是上面这样，那路径规划似乎有点问题（没有问题）
start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
start = [
    int(start_y * 100.0 / self.map_resolution - gx1),
    int(start_x * 100.0 / self.map_resolution - gy1),
]
然而可视化却是
curr_x, curr_y, curr_o, gy1, gy2, gx1, gx2 = sensor_pose
pos = (
    (curr_x * 100.0 / 5 - gx1)
    * 480
    / obstacle_map.shape[0],
    (obstacle_map.shape[1] - curr_y * 100.0 / 5 + gy1)
    * 480
    / obstacle_map.shape[1],
    np.deg2rad(-curr_o),
)
即使存在cv坐标系的区别，x 和 y以及gx和gy的相对变换关系应该是一样的，注意：两者 sensor_pose 赋值方式有些区别


## 各个split数据量大小
val: 1199
minival: 10
train: 


 ## bug
 - 为什么设置了vovabulary FULL，结果还是只有哪几类物体？object_nav_agent的prepocess obs进行了处理，已修复
 - 现在的ESC 如果用gt segmentation的话，结果可能是混乱的，因为 gt segmentation的id 和标签不知道，已解决
 - 语义图似乎有问题，语义图上看着是障碍物，但是还是走过去了
 - gaze 有bug， gaze 调用nav，但是如果gaze到的物体不是目标物体，就走近后发现不是，这时候会选择下一个目标，然后到下一个目标后，会跳过gaze

## git版本说明
1. init 最初版本
2. full semantic，能够建立所有recep的语义地图，并且已从llama3-8b中抽取物体共现关系
3. PSL alone, 单独增加 cyw/test/psl_agent.py 文件，并调试通
4. grounding sam 配置环境，加上grounding sam ,代码没怎么修改
5. place_before 修改 make_env 函数之前
6. place_data_collect 放置物体数据收集之前

## 潜在bug
1. 导航的mask好像是各个方向都可以走，这样会导致碰撞
2. obs的处理，goal_idx的处理，好像都只是单一样本，这样无法多线程运行？
```
obj_goal_idx, start_recep_idx, end_recep_idx = 1, 2, 3
```
3. 代码里面设置了 只要是esc导航，一定会建立full_vocab，即所有的容器都建立进去，但是消融实验值需要用room的时候，没有必要把其它容器也建立图

4. object 和 object 的 co-occur 矩阵，对角线元素全置为1，可能使得agent总是倾向于选择以前探索过的instance？


## 改进方向
1. 现在的psl选取，好像对，又好像不对？near-room maxtrix, near_obj matrix 难以确定是否正确；基于frontier的太过于离散了，或许使用聚类的方法？问题重点不在于此，可之后再做
2. 多进程，参考multiprocess, ray
3. 发现目标后，围着目标转一圈，也合理也不合理，在地图上是围着饶了一圈，但是并没有真正的在视角上饶一圈，没有窥探全貌
4. 从地图来看，recep是容易发现的，其实frontier并没有发生什么重大作用
5. 抓东西和放东西，仿真的视角非常不合理，什么都看不见了
6. fmm导航还是有些问题,隔着一堵墙,距离判断为很近


## 以为的bug
1. 现在的代码导航用的位姿是global位姿，但是map是local map, 这在p.global_downscaling=1的时候是允许的，因为global和local是一致的，不然就会发生错误，默认配置 global_downscaling 是 2
源代码：
```
start = [
   int(start_y * 100.0 / self.map_resolution - gx1),
   int(start_x * 100.0 / self.map_resolution - gy1),
]
```
又转换为了 local_pose, 所以是对的, 需要注意：语义图给出的local_pose 和 global_pose都是以m为单位，且 y 为第一轴的坐标值，x 为第二轴的坐标值

## done
1. 跑一下连续环境设置是什么结果？
2. 先用gt对比一下效果，目前来看，导航算法的停止点选择很不合理


## question
1. 为什么非gt 会这么大的影响rl place 效果？ rl有没有针对非gt训练
2. 为什么连续导航的实验结果不如离散导航的？
3. 导航里面发生碰撞算不算失败？

## idea
1. 选择目标点的时候考虑可交互性,拿取、放置(如果能训出来一个看到rgb，知道在什么位置交互，什么位置能看清物体，怎么找完大容器，那就很好)
2. 加上gaze at object
3. 重新训练place 策略
4. 加上场景布局匹配

5. 分割模型怎么着都要微调

6. 路径规划算法：[104] J. J. Kuffner and S. M. LaValle. Rrt-connect: An efficient approach to single-query path planning. In ICRA, 2000.

### 针对idea 1的进一步思考
1. 直接想让模型远远地看到物体，就知道哪些地方好交互，哪些地方不好交互可能太难了？或许让机器人走到东西附近，绕一圈，多输入一些信息，再预测哪些地方好交互，会比较简单
2. 即使要让机器人在很远的地方就预测可交互区域，也应该是一些序列迭代问题，第一步的预测接入到第二步中，再进行预测
3. 为什么不用VLM，在某些关键的时候调用（如看到可交互物体），区分出哪些点好交互，哪些点不好交互？
4. 或许预测的不应该是一个可交互点，而是各个点的affordance，或许应该按照规则性的方式，给定可交互区域，让模型打分，affordance,然后affordance分数随着运动过程不断迭代。（这样效果应该会更好一些，但是训练数据应该怎么来？）

## 环境配置版本说明
cyw/ovmm_env_20240513.yml 为配置yolo-world模型前的环境配置

## place 策略
鑫垚的放置策略:按照固定动作执行
```
        self.placing_actions = [
            DiscreteNavigationAction.MOVE_FORWARD,
            DiscreteNavigationAction.MANIPULATION_MODE,
            # ContinuousFullBodyAction([0,0,0,0,0,0,0,0,0,0], [0, -0.2, (6*3.14)/180.0]),
            ContinuousFullBodyAction([1,0,0,0,1,0,0,0,0,0], [0, 0, 0]),
            DiscreteNavigationAction.DESNAP_OBJECT,
            DiscreteNavigationAction.STOP
        ]
```
baseline 启发式策略
```
        if self.timestep < self.initial_orient_num_turns:
            if self.orient_turn_direction == -1:
                action = DiscreteNavigationAction.TURN_RIGHT
            if self.orient_turn_direction == +1:
                action = DiscreteNavigationAction.TURN_LEFT
            if self.verbose:
                print("[Placement] Turning to orient towards object")
        elif self.timestep < self.total_turn_and_forward_steps:
            if self.verbose:
                print("[Placement] Moving forward")
            action = DiscreteNavigationAction.MOVE_FORWARD
        elif self.timestep == self.total_turn_and_forward_steps:
            action = DiscreteNavigationAction.MANIPULATION_MODE
        elif self.timestep == self.t_go_to_top:
            # We should move the arm back and retract it to make sure it does not hit anything as it moves towards the target position
            action = self._retract(obs)
        elif self.timestep == self.t_go_to_place:
            if self.verbose:
                print("[Placement] Move arm into position")
            placement_height, placement_extension = (
                self.placement_voxel[2],
                self.placement_voxel[1],
            )

            current_arm_lift = obs.joint[4]
            delta_arm_lift = placement_height - current_arm_lift

            current_arm_ext = obs.joint[:4].sum()
            delta_arm_ext = (
                placement_extension
                - STRETCH_STANDOFF_DISTANCE
                - RETRACTED_ARM_APPROX_LENGTH
                - current_arm_ext
                + HARDCODED_ARM_EXTENSION_OFFSET
            )
            center_voxel_trans = np.array(
                [
                    self.placement_voxel[1],
                    self.placement_voxel[2],
                    self.placement_voxel[0],
                ]
            )
            delta_heading = np.rad2deg(get_angle_to_pos(center_voxel_trans))

            delta_gripper_yaw = delta_heading / 90 - HARDCODED_YAW_OFFSET

            if self.verbose:
                print("[Placement] Delta arm extension:", delta_arm_ext)
                print("[Placement] Delta arm lift:", delta_arm_lift)
            joints = np.array(
                [delta_arm_ext]
                + [0] * 3
                + [delta_arm_lift]
                + [delta_gripper_yaw]
                + [0] * 4
            )
            joints = self._look_at_ee(joints)
            action = ContinuousFullBodyAction(joints)
        elif self.timestep == self.t_release_object:
            # desnap to drop the object
            action = DiscreteNavigationAction.DESNAP_OBJECT
        elif self.timestep == self.t_lift_arm:
            action = self._lift(obs)
        elif self.timestep == self.t_retract_arm:
            action = self._retract(obs)
        elif self.timestep == self.t_extend_arm:
            action = DiscreteNavigationAction.EXTEND_ARM
        elif self.timestep <= self.t_done_waiting:
            if self.verbose:
                print("[Placement] Empty action")  # allow the object to come to rest
            action = DiscreteNavigationAction.EMPTY_ACTION
        else:
            if self.verbose:
                print("[Placement] Stopping")
            action = DiscreteNavigationAction.STOP

```
也是按照一套固定动作,但是这些量都是计算来的(然而交互点是一开始就选好了的)
```
            joints = np.array(
                [delta_arm_ext]
                + [0] * 3
                + [delta_arm_lift]
                + [delta_gripper_yaw]
                + [0] * 4
            )
            joints = self._look_at_ee(joints)
            action = ContinuousFullBodyAction(joints)
```
zxy的
```
 ContinuousFullBodyAction([1,0,0,0,1,0,0,0,0,0], [0, 0, 0]),
```