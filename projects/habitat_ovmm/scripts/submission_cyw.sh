#!/usr/bin/env bash

python projects/habitat_ovmm/eval_baselines_agent.py --evaluation_type $AGENT_EVALUATION_TYPE $@

# $@ 是其它位置参数
