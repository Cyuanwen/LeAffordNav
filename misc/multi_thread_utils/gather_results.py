"""
收集数据，并汇总
"""
import argparse
import os
import json
from typing import TYPE_CHECKING, Any, Dict, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import os
home_root = os.environ.get("HOME_ROBOT_ROOT")
import sys
sys.path.append(f"{home_root}/projects/habitat_ovmm/")
from utils.metrics_utils import get_stats_from_episode_metrics



def _aggregate_metrics(episode_metrics: Dict[str, Any]) -> Dict[str, float]:
    """Aggregates metrics tracked by environment."""
    aggregated_metrics = defaultdict(list)
    metrics = set(
        [
            k
            for metrics_per_episode in episode_metrics.values()
            for k in metrics_per_episode
            if k != "goal_name"
        ]
    )
    for v in episode_metrics.values():
        for k in metrics:
            if k in v:
                aggregated_metrics[f"{k}/total"].append(v[k])

    aggregated_metrics = dict(
        sorted(
            {
                k2: v2
                for k1, v1 in aggregated_metrics.items()
                for k2, v2 in {
                    f"{k1}/mean": np.mean(v1),
                    f"{k1}/min": np.min(v1),
                    f"{k1}/max": np.max(v1),
                }.items()
            }.items()
        )
    )

    return aggregated_metrics

def _summarize_metrics(episode_metrics: Dict) -> Dict:
    """Gets stats from episode metrics"""
    # convert to a dataframe
    episode_metrics_df = pd.DataFrame.from_dict(episode_metrics, orient="index")
    episode_metrics_df["start_idx"] = 0
    stats = get_stats_from_episode_metrics(episode_metrics_df)
    return stats

def _print_summary(summary: dict):
    """Prints the summary of metrics"""
    print("=" * 50)
    print("Averaged metrics")
    print("=" * 50)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",type=str,default='eval_hssd')
    parser.add_argument("--ep_num",type=int,default=10)
    parser.add_argument("--process_num",type=int,default=10)
    args = parser.parse_args()

    result_dir = os.path.join("datadump/results",args.exp_name)

    chunk_size = args.ep_num // args.process_num
    total_data = {}
    for i in range(args.process_num):
        # # debug
        # if i == 1 or i == 2 or i==3 or i==4:
        #     continue
        # if i != 1 and i != 2 and i!=3 and i!=4:
        #     continue
        if i!=args.process_num-1:
            start = i * chunk_size
            end = start + chunk_size
        else:
            start = i * chunk_size
            end = args.ep_num

        # 读取文件
        # datadump/results/eval_hssd/episode_results_0_1.json
        if not os.path.exists(os.path.join(result_dir,f"episode_results_{start}_{end}.json")):
            print(f"no this file for {i}")
            continue
        with open(os.path.join(result_dir,f"episode_results_{start}_{end}.json"),"r") as f:
            ep_results = json.load(f)
        if len(ep_results) == end - start:
            print(f"the file {i} is done")
        total_data = {
            **total_data,
            **ep_results
        }
    # 保存文件
    with open(os.path.join(result_dir,"episode_results.json"),"w") as f:
        json.dump(total_data,f)
    # 计算整体结果
    aggregated_metrics = _aggregate_metrics(total_data)
    # 保存整体结果
    # datadump/results/eval_hssd/aggregated_results_0_1.json
    with open(os.path.join(result_dir,"aggregated_results.json"),"w") as f:
        json.dump(aggregated_metrics,f)
    # summary results
    average_metrics = _summarize_metrics(total_data)
    _print_summary(average_metrics)
    


