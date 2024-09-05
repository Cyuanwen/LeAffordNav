### 开环执行效果不好的原因
1. 没有placement_point
![Alt text](image.png)
![Alt text](image-1.png)
![Alt text](image-2.png)
![Alt text](image-3.png)
感觉像是建图有问题一样，为什么明明没有end recep, 却建立起recep?
![Alt text](image-5.png)
![Alt text](image-6.png)
![Alt text](image-9.png) 认不出来，给不了mask，感觉这建图有问题？
![Alt text](image-15.png) 一开始认为那是chest_of_drawer，后来又认为不是
![Alt text](image-18.png) 离太近，一点没认出来，结果就没有放置点
![Alt text](image-20.png) 选不到一个放置点
![Alt text](image-21.png) 选不出来放置点
![Alt text](image-22.png) 选不出来放置点，建的图也有点问题，距离放置点还有挺远的


2. 放上去却判定碰撞

3. 没放到放置点
![Alt text](image-14.png) 按理说不应该隔这么远放不上去的

4. 认不出来放置容器
![Alt text](image-4.png)
![Alt text](image-7.png) couch 识别不出来，感觉不应该
![Alt text](image-10.png) 认不出来table
![Alt text](image-16.png) 选到一个很远的容器，进的容器认不出来

5. 隔着老远，判定为导航到了
![Alt text](image-8.png)
这语义图有点问题？或者nav2recep有点问题？

6. 一开始认出来一个容器，最后放的时候选的是另一个容器
![Alt text](image-11.png)
本来一开始选的是右手边这个

7. 放到边上，掉下去了
![Alt text](image-12.png)

8. 建图有问题，把路都挡住了
![Alt text](image-13.png)
一开始都识别到容器了
![Alt text](image-23.png) 还隔着一堵墙的距离，却认为导航到了？语义地图确实有点离谱

9. 放置点问题
![Alt text](image-17.png)
离太近，选的放置点很靠边

10. 离太近，选的放置点过于靠边，然后碰撞了
![Alt text](image-19.png)

11. 东西太大，掉下去了

## 总结
1. 建图似乎有点问题，明明离容器还差一截，但是建图却说已经到容器了
按理说看它代码没有近似，应该不会出现问题，难道是可视化的问题？要不就是nav2recep判定太早了
在发现目标的时候，只是判断朝向目标，就停止
```
            if self.discrete_actions:
                if relative_angle_to_closest_goal > 2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_closest_goal < -2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.STOP
```
不过，也有判断一个距离
stop = subset[self.du, self.du] < self.goal_tolerance
找容器和物体的时候会拒绝已经探索过的位置
```
    if reject_visited_regions:
        # remove the receptacles that the already been close to
        category_map = category_map * (
            1 - map_features[e, MC.BEEN_CLOSE_MAP, :, :]
        )
```
self.goal_tolerance  = 2.0 （2格）

2. 开环选择放置点确实不太合理，离得太近容易选到很靠边的放置点（不要迭代选择放置点，直接通过减掉机器人位移，计算准确的放置点）
3. 对于选不出来放置点的情况，基座点选取模型应该能够解决

有一个问题：在训练集上yolo效果已经和gt差不多，这时采集的数据还是和val有些差别，所以，考虑用val数据训练模型？


## gt失败的情况
1. 放在沙发靠背
![Alt text](image-24.png)
2. 确实存在伸的过长的情况
3. 放的高度太低，碰撞了
![Alt text](image-25.png)
![Alt text](image-26.png)

![Alt text](image-27.png) 难道是柜台太高了，伸手臂的时候也发生碰撞
按理说不应该存在这样的误差，所有东西都是严格计算出来的，难道高度估计错了？

![Alt text](image-28.png)
![Alt text](image-29.png)
为什么已经这么精细地调整了，还是放不到相应的位置？

![Alt text](image-30.png)
![Alt text](image-31.png)

![Alt text](image-32.png)
![Alt text](image-33.png)

4. 放的东西太大了

![Alt text](image-34.png)

![Alt text](image-35.png)