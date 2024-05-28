import os
import json


if __name__ == '__main__':

    extract_begin_index = 0
    extract_num = 50
    extract_end_index = extract_begin_index + extract_num
    src_json_file_path = '/root/qwen/Workplace/data/src/merge.json'
    dst_json_file_path = f'/root/qwen/Workplace/data/src/merge_extracted_{extract_begin_index}_{extract_end_index}.json'

    with open(src_json_file_path, 'r') as f:
        data = json.load(f)
    
    data_num = len(data)
    print(f'\nTotal data number: {data_num}\n')

    data = data[extract_begin_index:extract_end_index]

    with open(dst_json_file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f'\nProcessed data saved to {dst_json_file_path}\n')
        
