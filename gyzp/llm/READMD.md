使用LLM微调框架[hiyouga/LLaMA-Factory: Unify Efficient Fine-Tuning of 100+ LLMs (github.com)](https://github.com/hiyouga/LLaMA-Factory)。

## 安装

详细安装说明请参考[LLaMA-Factory/README_zh.md](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md#如何使用)。

```sh
# 创建python环境

conda create -n llama-factory python=3.10 -y
conda activate llama-factory


# 安装LLaMA-Factory

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[torch,metrics]
```

## 使用

本目录代码与配置文件均改自`LLaMA-Factory/examples`，将LLaMA-Factory中LLM的full-sft、lora-sft、predict的使用示例抽取出来，整理为独立的工作目录，便于使用。

### 快速开始

这里以Qwen1.5 7B的lora-sft和predict为例，说明如何使用LLaMA-Factory。可执行如下脚本。

```sh
# 下载Qwen1.5 7B模型
bash scripts/download_from_huggingface.sh

# 使用lora-sft微调Qwen1.5 7B
bash scripts/lora_sft.sh

# 使用predict微调Qwen1.5 7B
bash scripts/lora_predict.sh
```

### 数据集

微调与推理数据均在`data/dataset`目录下，要求在`data/dataset/dataset_info.json`文件中对数据集进行描述。

支持多种数据格式，详见`data/dataset/README_zh.md`。

### 配置文件

微调与推理的配置文件均在`config`目录下。修改微调和推理信息时，仅需修改配置文件，而无需修改`scripts`目录下的脚本（更改CUDA ID除外）。

例如，lora-sft对应的配置文件为`config/lora_sft.yaml`，predict对应的配置文件为`config/lora_predict.yaml`。常用参数含义已在配置文件中注释。

## Qwen1.5 7B 训练结果说明

### 微调模型说明

目录`temp/results/sft/qwen`下包含通义千问1.5 7B在两个任务中lora微调和full微调的结果。目录内容如下：

```sh
qwen
├── full-task1_prompt2_train  # 全量微调，仅输出output1，使用prompt2
├── full-task2_prompt0_train  # 全量微调，输出output1+output2，不加入prompt
├── lora-task1_prompt2_train  # lora微调，仅输出output1，使用prompt2
├── lora-task2_prompt0_train  # lora微调，输出output1+output2，不加入prompt
└── lora-task2_prompt3_train  # lora微调，输出output1+output2，使用prompt3
```

- task1、task2微调结果中，lora微调结果普遍优于full结果；不使用prompt（prompt0）的微调效果优于有prompt的微调结果；
- task2（output1+output2）微调结果中，`lora-task2_prompt0_train`效果最优，建议使用；


### 推理使用说明

修改`config/lora_predict.yaml`内容，如下所示：

1. 设置微调得到的`adapter_name_or_path`路径，如当前设置为最优的`lora-task2_prompt0_train`；
2. 设置推理输入数据集`dataset: manual_prompt0`，这里设置的`manual_prompt0`在`data/dataset/dataset_info.json`中已定义；
3. 设置推理结果输出路径`output_dir`，推理结束后将在此文件夹下找到如下内容：

    ```sh
    # 以如下推理结果为例
    temp/results/predict/qwen/lora-task2_prompt0_train-manual
    ├── all_results.json
    ├── generated_predictions.jsonl  # 推理结果文件
    └── predict_results.json
    ```
4. 上述得到的推理结果中，`output1和output2`仍混合在同一个字符串中（以句号作为分隔符）。可使用`utils/dataset/predict_postprocess_task2.py`进行后处理，处理结果将保存到`generated_predictions.jsonl`的路径下。例如：
    ```sh
    temp/results/predict/qwen/lora-task2_prompt0_train-manual
    ├── all_results.json
    ├── generated_predictions.jsonl  # 推理结果文件
    ├── predict_postprocessed.json   # 分割后的结果文件
    └── predict_results.json
    ```


```sh
### model
model_name_or_path: ./temp/models/Qwen1.5-7B  # base model path or huggingface model name
adapter_name_or_path: ./temp/results/sft/qwen/lora-task2_prompt0_train  # adapter path (such as the output of lora finetuning)

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
dataset_dir: ./data/dataset  # dataset path
dataset: manual_prompt0      # the name of the dataset being used, defined in dataset_dir/dataset_info.json
template: qwen               # see https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md#%E6%A8%A1%E5%9E%8B
cutoff_len: 512              # the maximum length of the input text
max_samples: 1000000         # the maximum number of samples to be used in the dataset
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ./temp/results/predict/qwen/lora-task2_prompt0_train-manual_prompt0-demo
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 1
predict_with_generate: true
```

注：推理可用单卡/多卡，需修改如下内容：
1. 推理脚本`scripts/lora_predict.sh`中，`cuda_visible_devices=0,1,2,3`字段；
2. 推理脚本`scripts/lora_predict.sh`中，指向的配置文件`accelerate_config_file_path=./config/accelerate_single_config.yaml`，修改卡数；


## LLM输出格式说明

因为llm输出只能是一个字符串，而不能是动作字符串列表，所以训练qwen时做了输出格式约定。

以输入“帮我拿一瓶水”为例：

```json
// 输入文件：/raid/home-robot/gyzp/llm/data/dataset/example.json
[
    {
        "instruction": "",
        "output": "",
        "input": "帮我拿一瓶水。"
    }
]
```

```json
// 推理输出文件：/raid/home-robot/gyzp/llm/temp/results/predict/qwen/lora-task2_prompt0_train-example/generated_predictions.jsonl
{"label": "", "predict": "寻找一瓶水；拿起水瓶。寻找物体, 一瓶水；拿起物体, 水瓶"}
```

注意：推理结果的输出文件为`jsonl`格式，且推理结果中，所有动作都包含在单个字符串中。下面需要按照约定的分割符处理以上输出，得到动作分隔的输出列表。

LLM推理输出字符串约定如下：

```
LLM输出：<output1>。<output2>
<output1>：<动作1>；<动作2>；<动作3> ...
<output2>: <动作1-谓语>，<动作1-目标物体>，<动作1-位置描述(如果有)>； <动作1-谓语>，<动作1-目标物体>，<动作1-位置描述(如果有)> ...
```

即，使用句号`。`分隔output1和output2，使用分号`；`分隔动作，使用逗号`，`分隔同一动作中的不同成分。


使用已写好的脚本`/raid/home-robot/gyzp/llm/utils/dataset/predict_postprocess_task2.py`，可以实现上述输出字符串的分割，分割结果示例如下：

```json
// 分割后的输出文件：/raid/home-robot/gyzp/llm/temp/results/predict/qwen/lora-task2_prompt0_train-example/predict_postprocessed.json
[
    {
        "instruction": "帮我拿一瓶水。",
        "output_1": [
            "寻找一瓶水",
            "拿起水瓶"
        ],
        "output_2": [
            "寻找物体, 一瓶水",
            "拿起物体, 水瓶"
        ]
    }
]
```

