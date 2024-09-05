#!/usr/bin/env bash
# 多进程
# 0 1   2   3   4   5   6   7   8   9
# 0 119 238 357 476 595 714 833 952 1071

# i=1
# i=2
# i=4
# i=6
# i=8
# i=9
# 定义数字列表
# numbers=(1 2 4 6 8 9)
numbers=(2 4 5 6 7 8 9)

# 获得 总 episode数
# python cyw/multi_thread_utils/get_ep_num.py $@
# total_episode=$EP_NUM
total_episode=$(cat cyw/temp/ep_num.txt)  
echo "**************"
echo "total episode number $total_episode"
# 计算每个进程应该处理的数据量
chunk_size=$((total_episode / 10))
echo "chunk size is $chunk_size"

# 使用 for 循环对数字列表进行迭代
for i in "${numbers[@]}"
do
    if [ $i -ne 9 ]; then
        start=$((i * chunk_size))
        end=$((start + chunk_size))
    else
        start=$((i * chunk_size))
        end=$total_episode
    fi
    # 在这里启动进程，并传递数据范围给每个进程
    echo "start process $i episode from $start to $end"
    # 这里可以根据实际需求启动相应的处理进程，比如调用另一个脚本或命令
    python projects/habitat_ovmm/eval_baselines_agent.py --evaluation_type $AGENT_EVALUATION_TYPE  --from_index $start --to_index $end $@ &
    # python cyw/test/test.py --from_index $start --to_index $end &
    sleep 30
done

# example
# bash projects/habitat_ovmm/scripts/submission_cyw_split.sh --EXP_NAME_suffix baseline
# bash projects/habitat_ovmm/scripts/submission_cyw_split.sh --agent_type gaze_place --baseline_config_path cyw/configs/agent/esc_yolo_gaze_place_cyw_agent.yaml --EXP_NAME_suffix hueristic_gaze

