# hueristic nav place policy
## success
![Alt text](image.png)
![Alt text](image-1.png)
![Alt text](image-2.png)
![Alt text](image-7.png)
![Alt text](image-8.png)
## fail
![Alt text](image-3.png)
![Alt text](image-4.png)
![Alt text](image-5.png)
![Alt text](image-6.png)


# huerisitic gaze place policy
## success

## fail
![Alt text](image-9.png)
看图像感觉差交互点差了一大截？
![Alt text](image-10.png)
![Alt text](image-11.png)
确实是差了很多，但是有时候又能够正确放置？

![Alt text](image-12.png)

![Alt text](image-13.png)
![Alt text](image-14.png)

![Alt text](image-15.png)
好像一侧着，这个算的就不准？

![Alt text](image-16.png)

![Alt text](image-17.png)

很迷，我感觉两个似乎看不出来很大的差别？
是不是确实离容器比较远

# hueristic nav place cyw
## success
![Alt text](image-18.png)
![Alt text](image-19.png)

## fail

![Alt text](image-20.png)
选的点是一个合理的点，但是机械臂不移动到该位置（难道机械臂已经达到最大长度了？）
似乎也不是因为没有朝向容器，所以不准确，有些情况，已经朝向容器了，放不下去，有些情况没有朝向容器，放下去了


### 最后放置的点在某些情况下会出现偏差？似乎距离计算或者哪里有问题
![Alt text](image-21.png)
![Alt text](image-22.png)
![Alt text](image-23.png)

![Alt text](image-24.png)
计算确实会存在一些失误

# nav orient to place 似乎有些不合理
ovmm_nav_to_place_succ is True
ovmm_nav_orient_to_place_succ is False

![Alt text](image-21.png)
像这样的情况会被判定为没有朝向容器，然后判定为失败

朝向容器是怎么计算的？

# place point and the final end point
![Alt text](image-25.png)
![Alt text](image-26.png)

![Alt text](image-27.png)
![Alt text](image-28.png)
难道是因为旋转角度离散和连续的问题？

![Alt text](image-29.png)
![Alt text](image-30.png)

![Alt text](image-31.png)
![Alt text](image-32.png)
这里计算似乎有点问题

![Alt text](image-33.png)
![Alt text](image-34.png)

![Alt text](image-35.png)
![Alt text](image-36.png)

# 有没有旋转偏执是否有差别
![Alt text](image-37.png)
![Alt text](image-38.png)
似乎改不改这个角度，无所谓？

## 将旋转角改为连续角度后，选择的placement point变离谱
![Alt text](image-39.png)

改变了偏向，为什么放置点似乎还是一点没变？
![Alt text](image-40.png)
原本不改变偏向的
![Alt text](image-41.png)

改为离散动作，根本没有执行？！

## 机械臂伸到最长，伸不过去了
![Alt text](image-42.png)

## yolo_detic 识别
![Alt text](image-44.png)
![Alt text](image-43.png)
为什么建出来的地图障碍物这么多？
把机器人卡的走不了

![Alt text](image-45.png)
![Alt text](image-46.png)

## 使用gt seg
![Alt text](image-47.png)
![Alt text](image-48.png)
似乎一样？
![Alt text](image-49.png)
![Alt text](image-50.png)

但是gt seg 确实不会出现卡在导航上的bug

所以，根本原因在于识别不出来？也不是
![Alt text](image-51.png)
![Alt text](image-52.png)
所以，为什么会卡住？




# trick
针对碰撞的问题，能不能先降低高度，然后再伸缩手臂？

改为开环执行，不知道为什么，总是有些离谱，总是有一些情况明显看着不对，明明放置点选的很合理，但是机械臂伸不到那个点
1. 达到机械臂最长限制
2. depth有误？
3. 中间某个计算过程有误？

## yolo seg 效果不好的原因
### 放错地方
![Alt text](image-53.png)

![Alt text](image-54.png) 真实在仿真环境里面那个沙发也叫chair，但机器人认不出来

![Alt text](image-55.png) 很莫名其妙，有些时候能认出来

![Alt text](image-57.png)
识别很不稳定：可能会出现走着走着，同一个东西，出现好几种不同的颜色

尽管最终都能找到一个容器，朝向它，可是会存在一些机械臂过长等问题，实际还是没有到放置点

![Alt text](image-58.png) 这里似乎是因为存在某些小bug？识别的是table,但是也直接在table前面停下来了

要是完全认不出来也行，偏偏就有些情况，看到一个角的时候能认出来，转过来又认不出来了
![Alt text](image-59.png)
![Alt text](image-60.png)

![Alt text](image-61.png)
上一步识别认为是couch, 这一步识别认为是 bed，已经要放置了，认不出来mask

子鹏采集的数据是不是有问题，离得远的很容易识别错？

## 一些实验结论
1. choose placement point 如果加上 check_visiblility 效果很差，且会报错，具体可看
cyw/datasets/place_dataset_debug/train/heuristic_agent_esc_yolo_gaze_place_cyw/_success.json
