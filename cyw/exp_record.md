# 实验记录
## 对比了修改place策略(让末端点再伸长一些,并且降低一些) 和原本的place策略,看起来还是原本的策略更好一些
## train 收集数据报错
```
agent_map_coord is [(113, 130)]
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████▏                        | 8/10 [01:53<00:28, 14.13s/it]
 25%|███████████████████████████████                                                                                             | 1/4 [02:38<07:54, 158.26s/it]
Traceback (most recent call last):
  File "/raid/home-robot/projects/habitat_ovmm/place_data_collection.py", line 683, in <module>
    gen_place_data(data_dir,dataset_file, env, agent, args.manual)
  File "/raid/home-robot/projects/habitat_ovmm/place_data_collection.py", line 447, in gen_place_data
    assert np.allclose(start_top_down_map_rot,start_agent_angle,rtol=0.01),"start_agent_angle is wrong"
AssertionError: start_agent_angle is wrong
(ovmm) tmn@drl-DGX-Station:/raid/home-robot$
```
收集到 7364 