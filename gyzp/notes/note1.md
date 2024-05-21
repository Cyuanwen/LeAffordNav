/raid/home-robot/datadump/results/eval_hssd_cyw_gtseg_print_img/episode_results.json


102817140_171
抓取失败
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/102817140_171/snapshot_326.png 抓取动作，但抓取失败
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/102817140_171/snapshot_345.png 放置动作

106878915_174887025_363
抓取失败
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/106878915_174887025_363/snapshot_324.png

105515211_173104185_951
抓取失败
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/105515211_173104185_951/snapshot_090.png 找到物体
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/105515211_173104185_951/snapshot_184.png 抓取，但没抓起来
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/105515211_173104185_951/snapshot_218.png 没抓起来物体，但继续找到了目标位置，做放置物体动作


107734176_176000019_11
没有抓取动作
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/107734176_176000019_11/snapshot_430.png 找到物体
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/107734176_176000019_11/snapshot_436.png 物体附近的一面墙壁，重复MOVEFORWARD，撞墙

107734176_176000019_63
没有抓取动作
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/107734176_176000019_63/snapshot_511.png
椅子只有一侧可以接近，似乎未找到合适的抓取点，机器人始终左右转动，没有抓取动作

106878915_174887025_380
没有抓取动作
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/106878915_174887025_380/snapshot_111.png
在凳子的一侧寻找合适的抓取点，左右转动，但始终没有抓取动作

102816756_625
没有抓取动作
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/102816756_625/snapshot_1127.png
在椅子的一侧寻找合适的抓取点，左右转动，但始终没有抓取动作

102816756_614
没有抓取动作
/raid/home-robot/datadump/images/eval_hssd_cyw_gtseg_print_img/102816756_614/snapshot_105.png
凳子有相邻两侧可以接近，但一侧距目标物体较远、另一侧较近，始终在较远的一侧移动寻找合适抓取位置，但没有移动到相邻的那一侧


8个找到物体、但没有成功抓取的例子。其中3个有抓取动作，但抓取失败；5个没有抓取动作，始终在目标物体周围移动。

有抓取动作但抓取失败的例子中，失败主要原因：距离过远

没有抓取动作，有如下几种情况：
1. 被墙壁阻挡，但仍重复“前进”动作；
2. 已经靠近目标物体，但重复左右旋转，几乎原地移动，没有找到合适抓取位置；
3. 不能找到距离目标物体较近的一侧，始终在较远的一侧移动；

---



