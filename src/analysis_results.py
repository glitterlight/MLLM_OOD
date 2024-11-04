import pandas as pd
import os
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix

ROOT_PATH = '/home/ma-user/work/code_dev/siming'


from src.utils.construct_inputs import *


def load_jsonl(path):
    results = []
    with open(path, 'r') as file:
        for line in file:
            results.append(json.loads(line))
    return results

def load_jsonl_dir(src_dir):
    jsonl_list = []
    file_name_list = list(os.listdir(src_dir))
    file_name_list.sort()
    for file_name in file_name_list:
        file_path = os.path.join(src_dir, file_name)
        if file_name.endswith('.jsonl'):
            jsonl_list.extend(load_jsonl(file_path))
    return jsonl_list

def extract_ood_label(org_text):
    if '[yes]' in org_text.lower():
        return 1
    elif '[no]' in org_text.lower():
        return 0
    else:
        return -1
    
def extract_results(gen_df):
    res_df = gen_df.copy()
    res_df['results'] = res_df['generation'].apply(extract_ood_label)
    return res_df

def deal_split_results(pred_list, group_num=5):
    split_preds_list = split_batch(pred_list, group_num)
    final_pred_list = [int(any(pred_list)) for pred_list in split_preds_list]

    return final_pred_list

if __name__ == '__main__':
    model_name = 'qwen2-vl-7b'
    id_dataset = 'waterbirds'
    ood_dataset = 'spurious_ood'
    method = 'domain_bird'
    group_num = 10 # class_size / group_size
    show_num = 20

    gen_result_name_list = [
        # (f'{ood_dataset}/{model_name}_labels-{id_dataset}_{method}', 0),
        (f'{id_dataset}/{model_name}_labels-{id_dataset}_{method}', 1),
    ]

    # print(gen_result_name_list)

    gen_result_dir = f'{ROOT_PATH}/ood_labels'

    pred_list = []
    true_list = []
    
    for gen_result_name, true_ood_labels in gen_result_name_list:
        gen_result_path = os.path.join(gen_result_dir, gen_result_name)
        gen_list = load_jsonl_dir(gen_result_path)
        gen_df = pd.DataFrame(gen_list)
        res_df = extract_results(gen_df)
        
        # print bad/good case
        cur_num = 0
        for line in res_df.iterrows():
            if line[1]['results'] == true_ood_labels:
            # if line[1]['results'] == -1:
                print(f'===================')
                print(line[1]['img_path'])
                print(line[1]['generation'])
                print(f'===================')

                cur_num += 1
                if show_num and cur_num >= show_num:
                    break

        tmp_pred_list = res_df['results'].tolist()
        if 'split' in method:
            tmp_pred_list = deal_split_results(tmp_pred_list, group_num)
            # print(f'tmp_pred_list: {tmp_pred_list}')
        pred_list.extend(tmp_pred_list)
        true_list.extend([true_ood_labels] * len(tmp_pred_list))
    
    # print(classification_report(true_list, pred_list))
    # print(confusion_matrix(true_list, pred_list))
