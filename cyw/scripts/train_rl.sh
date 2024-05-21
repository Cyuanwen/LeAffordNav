#/bin/bash

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

# ref projects/habitat_ovmm/README.md
set -x
# python -u -m habitat_baselines.run \
#    --exp-config habitat-baselines/habitat_baselines/config/ovmm/rl_skill.yaml \
#    --run-type train benchmark/ovmm=place \
#    habitat_baselines.checkpoint_folder=data/new_checkpoints/ovmm/place

python -u -m habitat_baselines.run \
    --config-name ovmm/rl_skill.yaml \
    habitat_baselines.evaluate=False \
    habitat_baselines.checkpoint_folder=data/new_checkpoints/ovmm/place
