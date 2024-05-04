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

## NOTE
1. 如增加了房间识别，可视化也要相应的修改


 ## info 
 visualize文件夹 src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/visualizer.py

 src/home_robot/home_robot/perception/constants.py 记录了很多种物体类别

 semmap 613行：voxels = du.splat_feat_nd(init_grid, feat, XYZ_cm_std).transpose(2, 3)
 把初始化的网格，语义特征，点云图结合起来，形成了体素图

 ## bug
 - 为什么设置了vovabulary FULL，结果还是只有哪几类物体？object_nav_agent的prepocess obs进行了处理

## git版本说明
1. init 最初版本
2. full semantic，能够建立所有recep的语义地图，并且已从llama3-8b中抽取物体共现关系
 

