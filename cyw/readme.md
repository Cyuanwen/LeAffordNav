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


 ## bug
 - 为什么设置了vovabulary FULL，结果还是只有哪几类物体？object_nav_agent的prepocess obs进行了处理，已修复
 - 现在的ESC 如果用gt segmentation的话，结果可能是混乱的，因为 gt segmentation的id 和标签不知道
 - 语义图似乎有问题，语义图上看着是障碍物，但是还是走过去了

## git版本说明
1. init 最初版本
2. full semantic，能够建立所有recep的语义地图，并且已从llama3-8b中抽取物体共现关系
3. PSL alone, 单独增加 cyw/test/psl_agent.py 文件，并调试通

## 潜在bug
1. 导航的mask好像是各个方向都可以走，这样会导致碰撞
2. obs的处理，goal_idx的处理，好像都只是单一样本，这样无法多线程运行？
```
obj_goal_idx, start_recep_idx, end_recep_idx = 1, 2, 3
```
3. 代码里面设置了 只要是esc导航，一定会建立full_vocab，即所有的容器都建立进去，但是消融实验值需要用room的时候，没有必要把其它容器也建立图


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

## TODO tomorro



 

