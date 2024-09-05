# 多线程测评
# # esc yol gaze place
export AGENT_EVALUATION_TYPE="local"
export HOME_ROBOT_ROOT=/raid/home-robot
export CUDA_VISIBLE_DEVICES=1,2,3
cd $HOME_ROBOT_ROOT
# conda activate ovmm
bash projects/habitat_ovmm/scripts/submission_cyw_multiprocess_escyolo.sh --EXP_NAME_suffix hueristic_gaze

## baseline
export AGENT_EVALUATION_TYPE="local"
export HOME_ROBOT_ROOT=/raid/home-robot
export CUDA_VISIBLE_DEVICES=1,2,3
cd $HOME_ROBOT_ROOT
conda activate ovmm
bash projects/habitat_ovmm/scripts/submission_cyw_multiprocess.sh --EXP_NAME_suffix baseline

# 默认val文件夹
# 文件默认存储在 datadump/results/eval_hssd 下

# NOTE 可以在 projects/habitat_ovmm/scripts/submission_cyw.sh 直接指定输入参数
# 也可以修改 projects/habitat_ovmm/scripts/submission_cyw.sh 脚本里面的运行命令

