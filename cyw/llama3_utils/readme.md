LLM常识数据生成
## 数据内容
1. 提示LLM的信息
2. LLM的返回
3. 解析后的目标物体和房间、物体的共现关系（包括正向关系，负向关系）
4. 以上内容结合后的最终内容（重要*）

## 数据生成流程
1. 使用cyw/llama3_utils/prompt_nagtive.py 或者cyw/llama3_utils/prompt_positive.py 生成相应的负向、正向关系提示以及获得LLM相应
2. 使用 cyw/llama3_utils/parse_response.py 获得解析LLM响应。 注意：目前版本可能需要手动修改个别回复
3. 使用 cyw/llama3_utils/aggregate_result.py 将结果聚合到一个文件里面