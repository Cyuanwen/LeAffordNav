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

本目录代码与配置文件均改自`/root/qwen/llm/LLaMA-Factory/examples`，将LLaMA-Factory中LLM的full-sft、lora-sft、predict的使用示例抽取出来，整理为独立的工作目录，便于使用。

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