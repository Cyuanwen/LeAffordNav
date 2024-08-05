# 提交说明
## docker 镜像配置
### 环境信息
docker version: 19.03.1
ubuntu version: lsb_release -a 18.04
### 准备
- pull 官方镜像
docker初始镜像：https://hub.docker.com/r/fairembodied/habitat-challenge/tags
docker pull fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023-ubuntu20.04
（NOTE 当前docker或者ubuntu版本太低，拉取高版本镜像后，在该镜像运行 不能开启python thread）

- 将本地代码提交到gitee
git push https://oauth2:5ecf0d713167fa16e98b8a01ba072536@gitee.com/drlchenyuanwen/home-robot.git
(该网址可用 git config --edit 编辑,密匙 5ecf0d713167fa16e98b8a01ba072536)

### 创建镜像
- 创建容器
（直接使用 projects/habitat_ovmm/docker/ovmm_baseline.Dockerfile 会报错 can't clone github，原因不明）

```
nvidia-docker run -it --mount type=bind,source=/raid/home-robot,target=/home-robot-cyw,readonly --net=host --shm-size=4g --name ovmm_baseline_submission_v4 fairembodied/habitat-challenge:homerobot-ovmm-challenge-2023-ubuntu20.04
```
- 配置容器
homerobot-ovmm-challenge-2023-ubuntu20.04 版本镜像 habitat-lab不是最新版本，homerobot-ovmm-challenge-2024-ubuntu22.04-v2 是一致的，但是服务器无法使用（需加--privileged，但这会导致配置的镜像在运行的时候也要加 --privileged ）
因此从gitee clone 项目，更新，重新安装一些包
1. clone home-robot项目
```
mkdir home-robot-v1
cd home-robot-v1
git clone https://oauth2:5ecf0d713167fa16e98b8a01ba072536@gitee.com/drlchenyuanwen/home-robot.git
```
2. 更新子模块，并重新安装包
```
. activate home-robot \
&& cd home-robot \
&& git submodule update --init --recursive src/third_party/detectron2 \
    src/home_robot/home_robot/perception/detection/detic/Detic \
    src/third_party/contact_graspnet \
&& pip install -e src/third_party/detectron2 \
&& pip install -r src/home_robot/home_robot/perception/detection/detic/Detic/requirements.txt \
&& pip install -e src/home_robot \

git submodule update --init --recursive src/third_party/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-lab
pip install -e src/third_party/habitat-lab/habitat-baselines
```
若报错无法找到某些模块：
1. 是否已安装该包 pip show <name> or conda list <name>
如未安装，则安装
2. 如已安装，确认版本；对于  pip insatall e .的包，确认指向是否正确
版本不对，重新安装

### 模型复制
```
mkdir -p home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models
cp /home-robot-cyw/src/home_robot/home_robot/perception/detection/detic/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth /home-robot-v1/home-robot/src/home_robot/home_robot/perception/detection/detic/Detic/models/



cd /home-robot-v1/home-robot/projects/habitat_ovmm
cp eval_baselines_agent.py /home-robot-v1/home-robot/projects/habitat_ovmm/agent.py
cp scripts/submission.sh /home-robot-v1/home-robot/submission.sh
```
将data进行软链接后，可以测试环境是否配置成功
bash submission.sh habitat.dataset.split=minival

### 将容器打包为镜像
- 打包初始镜像
上述环境测试成功后，可将容器打包为初始镜像
docker commit -m "ovmm_baseline_submission" -a "cyw" <container-name> <image-name>:<image-tag>
如：
```
docker commit -m "ovmm_baseline_submission" -a "cyw" ovmm_baseline_submission_v4_test ovmm_baseline_submission_v4:homerobot-ovmm-challenge-2023-ubuntu20.04
```
（NOTE:此镜像没有CMD，不会自动运行）
- 添加命令后的镜像
在 projects/habitat_ovmm/docker/ovmm_baseline_2024.Dockerfile 里面配置初始镜像，以及镜像启动后运行的命令
重新建立镜像：
```
 docker build . \
     -f docker/ovmm_baseline_2024.Dockerfile \
     -t ovmm_baseline_submission_v4_cmd \
     --network host
```
ovmm_baseline_submission_v4_cmd 为新镜像名称

### 测试镜像
```
./scripts/test_local_cyw.sh --docker-name ovmm_baseline_submission_v4_cmd --split minival
```

### tips
1. echo $HOSTNAME 可显示当前镜像名称
2. pip show <name> 可显示当前包的来源，依赖
3. docker run -it (交互) --rm (运行后删除) --name <container-name> ...
4. docker start <container-name> (启动容器)
5. docker exec -it <container-name> bash （执行容器命令，进入交互界面）
6. docker 版本更新：
    ```
        sudo apt-get update
        sudo apt-get upgrade docker-ce
    ```
    漫长等待...
7. docker images 显示镜像
8. docker ps (-a) 显示容器
9. docker rmi (删除镜像)
10. docker rm （删除容器）

### 总结
当前可用镜像：ovmm_baseline_submission_v4:homerobot-ovmm-challenge-2023-ubuntu20.04 (没加命令)
ovmm_baseline_submission_v4_cmd:latest(添加命令，提交使用)

### 代码修改后，如何提交
确保代码中所有用到的数据，模型，都在本项目文件下，代码中没有使用绝对路径。使用到的模型要传入到docker镜像里面（不能挂载）
1. 上传代码
``git push``
2. 若conda环境发生变动， 则基于镜像 ovmm_baseline_submission_v4:homerobot-ovmm-challenge-2023-ubuntu20.04 创建容器，在容器内配置好环境，再次提交镜像（可用容器 ovmm_baseline_submission_v4_test ）
```
nvidia-docker run -it --mount type=bind,source=/raid/home-robot,target=/home-robot-cyw,readonly --net=host --shm-size=4g --name ovmm_baseline_submission_v4_test ovmm_baseline_submission_v4:homerobot-ovmm-challenge-2023-ubuntu20.04
```
3. 若添加新的数据、模型，则基于镜像 ovmm_baseline_submission_v4:homerobot-ovmm-challenge-2023-ubuntu20.04 创建容器，将数据，模型复制到镜像的对应位置，再次提交镜像 （可用容器 ovmm_baseline_submission_v4_test）
复制模型，如
```
cp /home-robot-cyw/data/checkpoints /home-robot-v1/home-robot/data -r
cp /home-robot-cyw/data/default.physics_config.json /home-robot-v1/home-robot/data
cp /home-robot-cyw/data/models /home-robot-v1/home-robot/data -r
```
提交镜像
```
docker commit -m "ovmm_baseline_submission" -a "cyw" ovmm_baseline_submission_v4_test ovmm_baseline_submission_v4:homerobot-ovmm-challenge-2023-ubuntu20.04
```

4. 基于上述镜像，添加CMD，重新建立镜像
```
 docker build . \
     -f docker/ovmm_baseline_2024.Dockerfile \
     -t ovmm_baseline_submission_v4_cmd \
     --network host
```

5. 本地测试
./scripts/test_local_cyw.sh --docker-name ovmm_baseline_submission_v4_cmd --split minival

6. 提交
evalai push ovmm_baseline_submission_v4_cmd:latest --phase neurips-ovmm-test-standard-2023-2100 --private
（可参考https://eval.ai/web/challenges/challenge-page/2100/phases 、 docs/challenge.md）

### NOTE
代码中如果添加新的模型，数据，请在 projects/habitat_ovmm/docker/ovmm_baseline_2024.Dockerfile 里面 添加 docker镜像配置命令，并添加注释说明

conda 环境如果安装新 package 请在 projects/habitat_ovmm/docker/ovmm_baseline_2024.Dockerfile 里面 添加安装命令，并添加注释说明






