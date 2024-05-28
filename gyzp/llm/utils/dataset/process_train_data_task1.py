import os
import json

def merge_output_list_to_string(data):
    merge_field = 'output'
    delimeter = '；'

    print(f'Merging {merge_field} field to string with delimeter "{delimeter}"')

    for sample in data:
        sample[merge_field] = delimeter.join(sample[merge_field])
    return data

def get_prompt_from_file(file_list):
    prompt = ''
    for file in file_list:
        with open(file, 'r') as f:
            prompt += f.read()
    return prompt

def add_instruction(data):
    prompt_files = [
        '/root/qwen/Workplace/prompt/prompt2.txt'
    ]
    prompt = get_prompt_from_file(prompt_files)

    print(f'Adding instruction field with prompt from {prompt_files}')

    for sample in data:
        sample['input'] = sample['instruction']
        sample['instruction'] = prompt
        
    return data



if __name__ == '__main__':
    src_json_file_path = '/root/qwen/Workplace/data/src/all_ZH.json'
    dst_json_file_path = '/root/qwen/Workplace/data/dataset/all_ZH_prompt2.json'
    with open(src_json_file_path, 'r') as f:
        data = json.load(f)

    data = merge_output_list_to_string(data)
    data = add_instruction(data)

    with open(dst_json_file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f'Processed data saved to {dst_json_file_path}')
        
