import json
import os

def postprocess(src_jsonl_file_path, src_json_file_path, dst_json_file_path):
    '''
    @brief Postprocess the generated predictions and save the processed data to a json file
    @param src_jsonl_file_path: str, path to the jsonl file containing the generated predictions
    @param src_json_file_path: str, path to the json file containing the test data
    @param dst_json_file_path: str, path to save the postprocessed data
    '''
    with open(src_json_file_path, 'r') as f:
        label_data = json.load(f)
    
    with open(src_jsonl_file_path, 'r') as f:
        predict_data = [json.loads(line) for line in f]
    
    # assert len(label_data) == len(predict_data), 'Number of data in label and predict files do not match'

    data_num = min(len(label_data), len(predict_data))
    print(f'\nNumber of data: {data_num}\n')

    res = []
    
    for i in range(data_num):
        # sample = {
        #     'instruction': label_data[i]['input'],
        #     'predict_output_1': predict_data[i]['predict'].split('。')[0].split('；'),
        #     'label_output_1': label_data[i]['output'].split('。')[0].split('；'),
        #     'predict_output_2': predict_data[i]['predict'].split('。')[1].split('；'),
        #     'label_output_2': label_data[i]['output'].split('。')[1].split('；')
        # }
        sample = {
            'instruction': label_data[i]['input'],
            'output_1': predict_data[i]['predict'].split('。')[0].split('；'),
            'output_2': predict_data[i]['predict'].split('。')[1].split('；'),
        }
        res.append(sample)

    print(f'\nProcessed data sample:\n{res[0]}\n')
    
    os.makedirs(os.path.dirname(dst_json_file_path), exist_ok=True)
    with open(dst_json_file_path, 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    
    print(f'\nProcessed data saved to {dst_json_file_path}\n')

if __name__ == '__main__':
    src_jsonl_file_path = '/raid/home-robot/gyzp/llm/temp/results/predict/qwen/lora-task2_prompt0_train-example/generated_predictions.jsonl'
    src_json_file_path = '/raid/home-robot/gyzp/llm/data/dataset/example.json'

    # src_jsonl_file_path = '/root/qwen/Temp/results/predict/qwen/full-task2_prompt0_train-task2_prompt0_test/generated_predictions.jsonl'
    # src_json_file_path = '/root/qwen/Workplace/data/dataset/all_ZH_task2_prompt0_extracted_0_50.json'

    dst_json_file_path = os.path.join(os.path.dirname(src_jsonl_file_path), 'predict_postprocessed.json')
    
    postprocess(src_jsonl_file_path, src_json_file_path, dst_json_file_path)