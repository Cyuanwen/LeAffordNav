# 传输数据到另一个服务器
rsync -av --exclude='folder1' --exclude='folder2/' --exclude='*.jpg' /path/to/source/ user@remote_server:/path/to/destination/
--verbose


rsync -av --exclude-from '/raid/home-robot/.gitignore' /raid/home-robot cyw@172.18.41.151:~/ovmm


rsync -av -e 'ssh -p 13322' --exclude-from '/raid/home-robot/.gitignore' /raid/home-robot root@124.128.251.51:/cyw/data/

# docker 配置
# NOTE 容器里面的 /home-robot-v1 不要随意删除，环境基于这一文件配置
## 同步代码到容器
rsync -av --exclude-from '/home-robot-cyw/.gitignore' /home-robot-cyw /home-robot-v1
cd /home-robot-v1
mv home-robot-cyw home-robot
cd home-robot
mkdir data
## 说明： /home-robot-cyw 挂载了外部目录
## 复制数据，代码
cd /home-robot-cyw/data
cp cyw /home-robot-v1/home-robot/cyw/data -r
cp models /home-robot-v1/home-robot/data/models -r

cd /home-robot-cyw/cyw
cp data /home-robot-v1/home-robot/cyw -r

. activate home-robot \
&& cd /home-robot-v1/home-robot \
pip install -e src/third_party/habitat-lab/habitat-lab \
pip install -e src/third_party/habitat-lab/habitat-baselines

## 更新代码
rsync -av --exclude-from '/home-robot-cyw/.gitignore' /home-robot-cyw/ /home-robot-v1/home-robot/
## (不知道为什么habitat代码不会更新？)

## 测试docker
export HOME_ROBOT_ROOT=/home-robot-v1/home-robot/

## 可先在docker内部测试

## 将容器打包为镜像
docker commit -m "ovmm_baseline_submission" -a "cyw" ovmm_baseline_submission_v4_test ovmm_baseline_submission_v5:homerobot-ovmm-challenge-2023-ubuntu20.04

## 在镜像基础上添加命令
docker build . \
    -f docker/ovmm_baseline_v5.Dockerfile \
    -t ovmm_baseline_submission_v5_cmd \
    --network host

## 测试最终镜像
./scripts/test_local_cyw.sh --docker-name ovmm_baseline_submission_v5_cmd --split minival
# NOTE 这里使用的是默认配置

# 在另一个服务器上部署
rsync -av --exclude-from '/raid/home-robot/.gitignore' /raid/home-robot/ cyw@172.18.41.151:~/ovmm/home_robot/

# 提交上述基础版本
# https://eval.ai/web/challenges/challenge-page/2278/phases
evalai push ovmm_baseline_submission_v5_cmd:latest --phase homerobot-ovmm-test-standard-2024-2278 --private
# evalai push ovmm_baseline_submission_v5_cmd:latest --phase neurips-ovmm-test-standard-2023-2100 --private
# （连不上网，提交不了）
# Could not establish a connection to EvalAI. Please check the Host URL.
# ?!
# 所里服务器一大堆限制
# You haven't configured a Host URL for the CLI.
# The CLI would be using https://eval.ai as the default url.
# 无法访问evalai

# 下载到本地提交
docker save -o ovmm_baseline_submission_v5.tar ovmm_baseline_submission_v5

# tips
# 显示某个 id 的命令
ps -p 7105 -o cmd

# kill 某个进程

docker build . \
    -f docker/ovmm_baseline_v5.Dockerfile \
    -t ovmm_baseline_submission_v5_cmd \
    --network host