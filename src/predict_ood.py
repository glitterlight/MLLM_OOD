from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
import os
import time
import json
from loguru import logger
from multiprocessing import freeze_support
import pandas as pd
import argparse
import numpy as np
import math
import shutil
import gc
import random

from src.utils.construct_inputs import *


ROOT_PATH = '/home/ma-user/work/code_dev/siming'
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["OMP_NUM_THREADS"] = "32"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

random.seed(42)


def generate_text_prompt(classes, method='baseline'):
    text_prompt_dict = {
        'baseline': '''Does this figure belong to the following classes?\n{classes}\nAnswer with this format: "[yes]" or "[no]", and then explain the reason.''',

        'simple': '''Does this figure belong to the following classes?\n{classes}\nAnswer only with this format: "[yes]" or "[no]".''',

        'cot': '''Given classes list: {classes}\nPlease first answer which class the figure belongs to. Then compare it with the given classes respectively to analysis whether the class of the given figure belong to the given class. Finally, based on the analysis, you need to answer whether the figure belong to the given classes list with the format \"[yes]\" or \"[no]\".''',

        'cot2': '''Please first answer which class the figure belongs to. Then compare it with the given classes respectively to analysis whether the class of the given figure belong to the given class. Finally, based on the analysis, you need to answer whether the figure belong to the given classes list with the format \"[yes]\" or \"[no]\".\nGiven classes list: {classes}''',

        'cot3': '''Given classes list: {classes}\nPlease answer which class the figure belongs to. Then compare it with the given classes to anaswer whether the class of the given figure belong to the given class with the format \"[yes]\" or \"[no]\".''',

        'cot4': '''Please answer which class the figure belongs to. Then compare it with the given classes to anaswer whether the class of the given figure belong to the given class with the format \"[yes]\" or \"[no]\".\nGiven classes list: {classes}''',

        'longcot': '''Given classes list: {classes}\nPlease first answer which class the figure belongs to. Then refer to the given classes and analyze whether the class of the given figure belongs to the given class. Finally, based on the analysis, you need to answer whether the figure belongs to the given classes list with the format \"[yes]\" or \"[no]\".''',

        'longcot2': '''Please first answer which class the figure belongs to. Then refer to the given classes and analyze whether the class of the given figure belongs to the given class. Finally, based on the analysis, you need to answer whether the figure belongs to the given classes list with the format \"[yes]\" or \"[no]\".\nGiven classes list: {classes}''',

        'cot2_nofewshot': '''Please first answer which class the figure belongs to. Then compare it with the given classes respectively to analysis whether the class of the given figure belong to the given class. Finally, based on the analysis, you need to answer whether the figure belong to the given classes list with the format \"[yes]\" or \"[no]\".\nGiven classes list: {classes}''',

        'domain_bird': '''Please answer whether the figure belongs to the domain of bird with the format \"[yes]\" or \"[no]\", and then explain the reason.''',

        'domain_pet': '''Please answer whether the figure belongs to the domain of pet with the format \"[yes]\" or \"[no]\", and then explain the reason.''',

        'domain_car': '''Please answer whether the figure belongs to the domain of car with the format \"[yes]\" or \"[no]\", and then explain the reason.''',

        'domain_food': '''Please answer whether the figure belongs to the domain of food with the format \"[yes]\" or \"[no]\", and then explain the reason.''',
    }
    
    if 'split' not in method:
        text_prompts = [text_prompt_dict[method].format(classes=classes)]
    elif 'split' in method:
        base_method, _, group_size = method.split('_')
        group_size = int(group_size)
        logger.debug(f'split group size: {group_size}')
        classes_list = split_batch(classes, group_size)
        text_prompts = [text_prompt_dict[base_method].format(classes=classes) for classes in classes_list]

    return text_prompts


def generate_batch_outputs(llm, sampling_params, llm_inputs_list, img_path_list, 
                           dst_path, batch_size=1000, save_flag=True):
    output_text_list = []
    if not batch_size:
        logger.debug(f'batch size setting disabled')
        batch_size = len(img_path_list)
    
    logger.debug(f'currrent batch size: {batch_size}')
    batch_inputs_list = split_batch(list(zip(img_path_list, llm_inputs_list)), batch_size)
    logger.debug(f'batch number: {len(batch_inputs_list)}')
    for i, inputs_list in enumerate(batch_inputs_list):
        cur_img_path_list, cur_llm_inputs = zip(*inputs_list)
        # logger.debug(cur_img_path_list)
        outputs = llm.generate(cur_llm_inputs, sampling_params=sampling_params)
        output_texts = [output.outputs[0].text for output in outputs]
        output_text_list.extend(output_texts)
        if save_flag:
            save_results(cur_img_path_list, output_texts, dst_path)
            st_idx = i * batch_size
            ed_idx = st_idx + len(cur_img_path_list) - 1
            logger.debug(f'batch number: {i}, local idx {st_idx} to {ed_idx} results have been saved to {dst_path}.')

    return output_text_list


def save_results(img_path_list, output_texts, dst_path):
    res_list = []
    with open(dst_path, 'a', encoding='utf-8') as f:
        for img_path, output_text in zip(img_path_list, output_texts):
            # logger.debug(f'type of output_text: {type(output_text)}')
            res = {
                'img_path': img_path,
                'generation': output_text
            }
            f.write(json.dumps(res, ensure_ascii=False))
            f.write('\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='baseline')
    parser.add_argument('--model_name', type=str, default='qwen2-vl-7b')
    parser.add_argument('--classes_dataset_name', type=str, default='imagenet10')
    parser.add_argument('--src_dir', type=str, default=f'{ROOT_PATH}/datasets/imagenet10/val')
    parser.add_argument('--dst_dir', type=str, default=f'{ROOT_PATH}/ood_labels/imagenet10')
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--few_shot_path', type=str, default=None)
    parser.add_argument('--deal_num', type=int, default=None)
    parser.add_argument('--save_flag', type=int, default=1)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()
    logger.debug(f'args:\n{args}')

    # clean memory cache
    torch.cuda.empty_cache()
    gc.collect()


    model_dir = f'{ROOT_PATH}/models'
    model_name = args.model_name
    model_path = os.path.join(model_dir, model_name)

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    
    classes_path = f'{ROOT_PATH}/data/classes.json'
    classes_dataset_name = args.classes_dataset_name

    os.makedirs(dst_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(classes_path, 'r') as f:
        classes_dict = json.load(f)
    
    text_prompts = generate_text_prompt(classes_dict[classes_dataset_name], method=args.method)
    logger.debug(f'text_prompts: {text_prompts}')
    logger.debug(f'len(text_prompts): {len(text_prompts)}')

    img_path_list = get_img_path_list(src_dir, args.deal_num)
    total_num = len(img_path_list)
    logger.debug(f'total image number: {total_num}')


    # assign tasks for workers
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    logger.debug(f'rank: {rank}')
    logger.debug(f'world_size: {world_size}')

    # path of current worker generate outputs
    cur_dst_dir = os.path.join(dst_dir, f'{model_name}_labels-{classes_dataset_name}_{args.method}')
    if rank == 0 and os.path.exists(cur_dst_dir) and os.path.isdir(cur_dst_dir) and os.listdir(cur_dst_dir):
        shutil.rmtree(cur_dst_dir)
    os.makedirs(cur_dst_dir, exist_ok=True)
    rank_str = str(rank).zfill(3)
    dst_path = os.path.join(cur_dst_dir, f'{rank_str}.jsonl')

    img_path_list = np.array_split(img_path_list, world_size)[rank]
    mean_worker_deal_size = math.ceil(total_num / world_size)
    st_idx = rank * mean_worker_deal_size
    ed_idx = st_idx + len(img_path_list) - 1
    
    logger.debug(f'this worker handle {len(img_path_list)} images, from idx {st_idx} to {ed_idx}')



    few_shot_msgs, few_shot_num = None, 0
    if args.few_shot_path:
        few_shot_msgs, few_shot_num = construct_few_shot_msgs(args.few_shot_path)
    # logger.debug(f'few_shot_msgs: {few_shot_msgs}')
    
    # img_path_list = ['/home/ma-user/work/code_dev/siming/datasets/iNaturalist/images/0eba707ea91a73f899c6105471b89d70.jpg']
    messages_list = construct_text_inputs(img_path_list, text_prompts, backbone='vllm', few_shot_msgs=few_shot_msgs)
    # logger.debug(messages_list[0])
    logger.debug(f'number of input messages: {len(messages_list)}')
    
    # generate inputs for mllm
    processor = AutoProcessor.from_pretrained(model_path)
    llm_inputs_list = construct_llm_inputs(messages_list, processor, backbone='vllm', few_shot_num=few_shot_num)
    logger.debug(f'number of llm requests: {len(llm_inputs_list)}')
    
    # load model
    freeze_support()
    st_time = time.time()
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        tensor_parallel_size=args.num_workers,
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        seed=42
    )

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_tokens=1024,
        stop_token_ids=[],
    )

    ed_time = time.time()
    logger.debug(f"[INFO] model load time: {ed_time - st_time}")

    #generate outputs
    if 'split' in args.method:
        base_method, _, group_size = args.method.split('_')
        group_size = int(group_size)
        group_num = math.ceil(len(classes_dict[classes_dataset_name]) / group_size)
        logger.debug(f'split group size: {group_size}, split group num: {group_num}')
        related_img_path_list = []
        for img_path in img_path_list:
            related_img_path_list.extend([img_path] * group_num)
        related_img_path_list = related_img_path_list[:len(llm_inputs_list)]
    else:
        related_img_path_list = img_path_list

    st_time = time.time()
    output_text_list = generate_batch_outputs(llm, sampling_params, llm_inputs_list, related_img_path_list, dst_path,
                                              batch_size=args.batch_size, save_flag=args.save_flag)
    ed_time = time.time()
    logger.debug(f"[INFO] model generate time: {ed_time - st_time}")

    # show outputs
    show_num = 20
    for i, (text, img_path) in enumerate(zip(output_text_list[:show_num], related_img_path_list[:show_num])):
        print(f'===================')
        print(f'number: {i}')
        print(f'img path: {img_path}')
        print(text)
        print(f'===================')


    