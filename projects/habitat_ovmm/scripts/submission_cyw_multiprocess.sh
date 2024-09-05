#!/usr/bin/env bash
# 多进程

# 获得 总 episode数
python cyw/multi_thread_utils/get_ep_num.py $@
# total_episode=$EP_NUM
total_episode=$(cat cyw/temp/ep_num.txt)  
echo "**************"
echo "total episode number $total_episode"
# 计算每个进程应该处理的数据量
chunk_size=$((total_episode / 10))
echo "chunk size is $chunk_size"

# 使用循环启动10个进程，并将数据范围传递给每个进程
for ((i=0; i<10; i++)); do
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


# python projects/habitat_ovmm/eval_baselines_agent.py --evaluation_type $AGENT_EVALUATION_TYPE $@

# 等待所有进程启动完成
wait

echo "All processes have ended."

# 获得exp name
exp_name='eval_hssd'
for ((i=0; i<$#; i++)); do
    eval arg=\$$i
    # echo  "arg is $arg"
    if [[ $arg == --EXP_NAME_suffix* ]]; then
        j=$((i + 1))
        eval suffix=\$$j
        # echo "j is $j, and i is $i"
        # echo "suffix is $suffix"
        exp_name="eval_hssd_$suffix"
    fi
done
echo "exp name is $exp_name"


echo "gather data ======================"
python cyw/multi_thread_utils/gather_results.py --exp_name $exp_name --ep_num $total_episode --process_num 10

# 运行示例
# example
# bash projects/habitat_ovmm/scripts/submission_cyw_multiprocess.sh habitat.dataset.split=minival
