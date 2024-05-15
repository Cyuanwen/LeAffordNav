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

## 各个split数据量大小
val: 1199
minival: 10
train: 


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

4. object 和 object 的 co-occur 矩阵，对角线元素全置为1，可能使得agent总是倾向于选择以前探索过的instance？


## 改进方向
1. 现在的psl选取，好像对，又好像不对？near-room maxtrix, near_obj matrix 难以确定是否正确；基于frontier的太过于离散了，或许使用聚类的方法？问题重点不在于此，可之后再做
2. 多进程，参考multiprocess, ray
3. 发现目标后，围着目标转一圈，也合理也不合理，在地图上是围着饶了一圈，但是并没有真正的在视角上饶一圈，没有窥探全貌
4. 从地图来看，recep是容易发现的，其实frontier并没有发生什么重大作用


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
1. 跑一下连续环境设置是什么结果？
2. 先用gt对比一下效果，目前来看，导航算法的停止点选择很不合理


## question
1. 为什么非gt 会这么大的影响rl place 效果？ rl有没有针对非gt训练

## idea
1. 选择目标点的时候考虑可交互性,拿取、放置
2. 加上gaze at object
3. 重新训练place 策略
4. 加上场景布局匹配

5. 分割模型怎么着都要微调

## 环境配置版本说明
cyw/ovmm_env_20240513.yml 为配置yolo-world模型前的环境配置

