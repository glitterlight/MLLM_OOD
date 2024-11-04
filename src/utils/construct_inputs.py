from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
import os
import time
import json
import numpy as np
import math
import shutil
from loguru import logger
import random


def get_img_path_list(img_dir, num=None):
    img_exts = ['.jpg', '.jpeg', '.png']

    img_path_list = []

    for root, dirs, files in os.walk(img_dir):
        for file in files:
            # 检查文件扩展名是否为图片格式
            if os.path.splitext(file)[1].lower() in img_exts:
                # 将图片的完整路径添加到列表中
                img_path_list.append(os.path.join(root, file))

    if num:
        random.shuffle(img_path_list)
        img_path_list = img_path_list[:num]
    
    return img_path_list

def construct_few_shot_msgs(few_shot_path):
    with open(few_shot_path, 'r') as f:
        few_shot_dict = json.load(f)
    return few_shot_dict['content'], few_shot_dict['num']


def construct_vllm_prompt(img_path, text, few_shot_msgs=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    if few_shot_msgs:
        messages.extend(few_shot_msgs)
    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path,
                "min_pixels": 224 * 224,
                "max_pixels": 1280 * 28 * 28,
            },
            {"type": "text", "text": text},
        ]
    }
    messages.append(user_msg)

    return messages

def construct_hf_prompt(img_path, text, few_shot_msgs=None):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    if few_shot_msgs:
        messages.extend(few_shot_msgs)
    
    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path,
            },
            {"type": "text", "text": text},
        ]
    }
    messages.append(user_msg)

    return messages

def construct_text_inputs(img_path_list, texts, backbone='vllm', few_shot_msgs=None):
    messages_list = []

    for img_path in img_path_list:
        for cur_text in texts:
            if backbone == 'vllm':
                message = construct_vllm_prompt(img_path, cur_text, few_shot_msgs)
            elif backbone == 'hf':
                message = construct_hf_prompt(img_path, cur_text, few_shot_msgs)
            messages_list.append(message)

    return messages_list

def split_batch(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def construct_llm_inputs(messages_list, processor, backbone='vllm', few_shot_num=0):

    prompts = [processor.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
    ) for msg in messages_list]

    # print(f'prompts: {prompts}', flush=True)
    # print(f"few_shot_num: {few_shot_num}")

    image_inputs, video_inputs = process_vision_info(messages_list)
    # logger.debug(f'len(image_inputs): {len(image_inputs)}')
    
    
    if backbone == 'vllm':
        llm_inputs_list = []
        batch_image_inputs = split_batch(image_inputs, few_shot_num+1)
        for i, (prompt, image_inputs) in enumerate(zip(prompts, batch_image_inputs)):

            mm_data = {}
            if image_inputs is not None:
                # mm_data["image"] = [image_inputs]
                mm_data["image"] = image_inputs
                # mm_data["video"] = None

            llm_inputs = {
                'prompt': prompt,
                'multi_modal_data': mm_data,
            }
            llm_inputs_list.append(llm_inputs)
        return llm_inputs_list
    
    elif backbone == 'hf':
        inputs = processor(
            text=prompts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    else:
        return None

def construct_hf_inputs(messages_list, processor):
    prompts = [processor.apply_chat_template(
        msg,
        tokenize=False,
        add_generation_prompt=True,
    ) for msg in messages_list]

    image_inputs, video_inputs = process_vision_info(messages_list)
    llm_inputs_list = []

    for prompt, image_input in zip(prompts, image_inputs):

        mm_data = {}
        if image_input is not None:
            mm_data["image"] = [image_input]

        llm_inputs = {
            'prompt': prompt,
            'multi_modal_data': mm_data,
        }
        llm_inputs_list.append(llm_inputs)
    
    return llm_inputs_list