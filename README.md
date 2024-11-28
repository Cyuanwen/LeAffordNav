# Experiments to run (Ordered by priority)
Please provide the results of those experiments for the test-standard and test-challenge (if convenient) split.
1. escNav+PointUnetGaze_v2 (Must-Have)
2. escNav+PointUnetGaze_v1 (Must-Have)
3. escNav (Must-Have)
4. PointUnetGaze_v2 (Nice-to-Have)
5. PointUnetGaze_v1 (Nice-to-Have)

## Create the enviroment
The file [home-robot.yaml](home-robot.yaml) contains the configuration for the Conda environment. Overall, the environment matches the configuration of the original home-robot project, with a few differences:

1. Added the ultralytics package for YOLO.
2. I have modified the code in src/home_robot, so you need to reinstall this sub-project by 
```
cd $HOME_ROBOT_ROOT/src/home_robot
conda activate home_robot
pip install -e .
```

## Link data
Link your home-robot data to [data](data)

## Download the model
Download the model use command (recomand)
```
pip install gdown
gdown https://drive.google.com/uc?id=1VzjU-P-dYp2_4Mnh_HsmJq6vng916NoC
```

Or, you can download the model through browser: 
https://drive.google.com/file/d/1VzjU-P-dYp2_4Mnh_HsmJq6vng916NoC/view?usp=sharing

Put the model folder in the root directory of the home-robot project.

# Set the environment variable
Before running the code, make sure the environment variables are set correctly, such as:
```
export AGENT_EVALUATION_TYPE="local"
export HOME_ROBOT_ROOT=/home-robot-commit
export CUDA_VISIBLE_DEVICES=1,2,3
cd $HOME_ROBOT_ROOT
conda activate home-robot
```

In the following section, we introduce the submission script and provide two methods for evaluating the proposed approach.

**Method 1** involves running 5 processes simultaneously. While this method is slightly inconvenient, it offers significantly faster evaluation.

**Method 2** runs a single process for evaluation, which is slower but simpler to manage.

You can choose either method based on your preferences. However, I recommend *Method 1* if you have sufficient GPU resources.

After introduce the scripts, I will provide the commands needed to run my experiments, which you can use with just a few modifications.

# Running Multiple Process Submission Scripts (recommended)
This guide provides instructions for running multiple process submission scripts in the Home-Robot project. In this approach, all episodes are divided into five subsets, with each subset processed by a separate process. This method allows us to obtain results more quickly.
## 1. Run the Main Submission Script
First, run the main submission script:
```
bash projects/habitat_ovmm/scripts/submission_multiprocess.sh --agent_type gaze_place --baseline_config_path {config_path} --EXP_NAME_suffix {exp_name} habitat.dataset.split=test
```

Results will be saved in f"datadump/results/eval_hssd_{exp_name}". To check the results, run:
```
python misc/multi_thread_utils/gather_results.py --exp_name eval_hssd_{exp_name} --ep_num {total_episode} --process_num 5
```
Here, {total_episode} refers to the number of episodes in this split (e.g., 1199 for validation, 10 for minival). After running this command, you will see output like:
```
no this file for 0
no this file for 1
the file 2 is done
the file 3 is done
no this file for 4
==================================================
Averaged metrics
==================================================
episode_count: 4.0
does_want_terminate: 0.25
num_steps: 827.25
find_object_phase_success: 0.5
pick_object_phase_success: 0.5
find_recep_phase_success: 0.5
overall_success: 0.25
partial_success: 0.4375
==================================================
```
In the output, "no this file for 0" indicates that process 0 did not start correctly, while "the file 2 is done" means that process 2 completed successfully.

## 2. Restart Failed Processes
If some processes (e.g., 0, 1, 4) did not start correctly (often due to GPU limitations), you can restart them with:
```
bash projects/habitat_ovmm/scripts/submission_multiprocess_split.sh {chunks ID} --agent_type gaze_place --baseline_config_path {config_path} --EXP_NAME_suffix {exp_name} habitat.dataset.split=test
```

## Running Commands
This section provides commands to run experiments. They have been tested locally and can be executed directly.

Note: In the following commands, please replace "test" with the actual name of the test split.

### 1. escNav+PointUnetGaze_v2 (Must-Have)

**[Set environment variable](#set-the-environment-variable)**
```
export AGENT_EVALUATION_TYPE="local"
export HOME_ROBOT_ROOT=/home-robot-commit
export CUDA_VISIBLE_DEVICES=1,2,3
cd $HOME_ROBOT_ROOT
conda activate home-robot
```
**step 1**: run the main submission script
```
bash projects/habitat_ovmm/scripts/submission_multiprocess.sh --agent_type gaze_place --baseline_config_path config/agent/escNav_PointUnetGaze_v2.yaml --EXP_NAME_suffix escNav_PointUnetGaze_v2 habitat.dataset.split=test
``` 
**step 2**: check the result
```
python misc/multi_thread_utils/gather_results.py --exp_name eval_hssd_escNav_PointUnetGaze_v2 --ep_num {total_episode} --process_num 5
```
**step 3**: Restart any processes that did not run correctly (if any), such as:
```
bash projects/habitat_ovmm/scripts/submission_multiprocess_split.sh {ids} --agent_type gaze_place --baseline_config_path config/agent/escNav_PointUnetGaze_v2.yaml --EXP_NAME_suffix escNav_PointUnetGaze_v2 habitat.dataset.split=test
```
Note: Replace {ids} with the thread IDs that failed (e.g., "3 4") such as:
```
bash projects/habitat_ovmm/scripts/submission_multiprocess_split.sh 3,4 --agent_type gaze_place --baseline_config_path config/agent/escNav_PointUnetGaze_v2.yaml --EXP_NAME_suffix escNav_PointUnetGaze_v2 habitat.dataset.split=test
```
and the first thread ID is 0.

Tips: you can check the results three or four hours after you run the main submission command, and re-run those processes that did not run correctly.

### 2. escNav+PointUnetGaze_v1 (Must-Have)

**[Set environment variable](#set-the-environment-variable)** as before. Note that you should specify the available GPUs by using `export CUDA_VISIBLE_DEVICES=2,3`.

**step 1**: run the main submission script
```
bash projects/habitat_ovmm/scripts/submission_multiprocess.sh --agent_type gaze_place --baseline_config_path config/agent/escNav_PointUnetGaze_v1.yaml --EXP_NAME_suffix escNav_PointUnetGaze_v1 habitat.dataset.split=test
``` 
**step 2**: check the result
```
python misc/multi_thread_utils/gather_results.py --exp_name eval_hssd_escNav_PointUnetGaze_v1 --ep_num {total_episode} --process_num 5
```
**step 3**: Restart any processes that did not run correctly (if any), such as:
```
bash projects/habitat_ovmm/scripts/submission_multiprocess_split.sh {ids} --agent_type gaze_place --baseline_config_path config/agent/escNav_PointUnetGaze_v1.yaml --EXP_NAME_suffix escNav_PointUnetGaze_v1 habitat.dataset.split=test
```

### 3. escNav (Must-Have)
**[Set environment variable](#set-the-environment-variable)** as before. Note that you should specify the available GPUs by using `export CUDA_VISIBLE_DEVICES=2,3`.

**step 1**: run the main submission script
```
bash projects/habitat_ovmm/scripts/submission_multiprocess.sh --agent_type gaze_place --baseline_config_path config/agent/escNav.yaml --EXP_NAME_suffix escNav habitat.dataset.split=test
``` 
**step 2**: check the result
```
python misc/multi_thread_utils/gather_results.py --exp_name eval_hssd_escNav --ep_num {total_episode} --process_num 5
```
**step 3**: Restart any processes that did not run correctly (if any), such as:
```
bash projects/habitat_ovmm/scripts/submission_multiprocess_split.sh {ids} --agent_type gaze_place --baseline_config_path config/agent/escNav.yaml --EXP_NAME_suffix escNav habitat.dataset.split=test
```

### 4. PointUnetGaze_v2 (Nice-to-Have)
**[Set environment variable](#set-the-environment-variable)** as before. Note that you should specify the available GPUs by using `export CUDA_VISIBLE_DEVICES=2,3`.

**step 1**: run the main submission script
```
bash projects/habitat_ovmm/scripts/submission_multiprocess.sh --agent_type gaze_place --baseline_config_path config/agent/PointUnetGaze_v2.yaml --EXP_NAME_suffix PointUnetGaze_v2 habitat.dataset.split=test
``` 
**step 2**: check the result
```
python misc/multi_thread_utils/gather_results.py --exp_name eval_hssd_PointUnetGaze_v2 --ep_num {total_episode} --process_num 5
```
**step 3**: Restart any processes that did not run correctly (if any), such as:
```
bash projects/habitat_ovmm/scripts/submission_multiprocess_split.sh {ids} --agent_type gaze_place --baseline_config_path config/agent/PointUnetGaze_v2.yaml --EXP_NAME_suffix PointUnetGaze_v2 habitat.dataset.split=test
```

### 5. PointUnetGaze_v1 (Nice-to-Have)
**[Set environment variable](#set-the-environment-variable)** as before. Note that you should specify the available GPUs by using `export CUDA_VISIBLE_DEVICES=2,3`.

**step 1**: run the main submission script
```
bash projects/habitat_ovmm/scripts/submission_multiprocess.sh --agent_type gaze_place --baseline_config_path config/agent/PointUnetGaze_v1.yaml --EXP_NAME_suffix PointUnetGaze_v1 habitat.dataset.split=test
``` 
**step 2**: check the result
```
python misc/multi_thread_utils/gather_results.py --exp_name eval_hssd_PointUnetGaze_v1 --ep_num {total_episode} --process_num 5
```
**step 3**: Restart any processes that did not run correctly (if any), such as:
```
bash projects/habitat_ovmm/scripts/submission_multiprocess_split.sh {ids} --agent_type gaze_place --baseline_config_path config/agent/PointUnetGaze_v1.yaml --EXP_NAME_suffix PointUnetGaze_v1 habitat.dataset.split=test
```

# Run the code with single thread
You can run the code with a single thread, which may be slower. After [set environment variable](#set-the-environment-variable), use the following commands to run the evaluation.
### 1. escNav+PointUnetGaze_v2 (Must-Have)
```
bash projects/habitat_ovmm/scripts/submission.sh --agent_type gaze_place --baseline_config_path config/agent/escNav_PointUnetGaze_v2.yaml --EXP_NAME_suffix escNav_PointUnetGaze_v2 habitat.dataset.split=test
```
### 2. escNav+PointUnetGaze_v1 (Must-Have)
```
bash projects/habitat_ovmm/scripts/submission.sh --agent_type gaze_place --baseline_config_path config/agent/escNav_PointUnetGaze_v1.yaml --EXP_NAME_suffix escNav_PointUnetGaze_v1 habitat.dataset.split=test
```
### 3. escNav (Must-Have)
```
bash projects/habitat_ovmm/scripts/submission.sh --agent_type gaze_place --baseline_config_path config/agent/escNav.yaml --EXP_NAME_suffix escNav habitat.dataset.split=test
```
### 4. PointUnetGaze_v2 (Nice-to-Have)
```
bash projects/habitat_ovmm/scripts/submission.sh --agent_type gaze_place --baseline_config_path config/agent/PointUnetGaze_v2.yaml --EXP_NAME_suffix PointUnetGaze_v2 habitat.dataset.split=test
```
### 5. PointUnetGaze_v1 (Nice-to-Have)
```
bash projects/habitat_ovmm/scripts/submission.sh --agent_type gaze_place --baseline_config_path config/agent/PointUnetGaze_v1.yaml --EXP_NAME_suffix PointUnetGaze_v1 habitat.dataset.split=test
```