import argparse
import os
from loguru import logger


ROOT_PATH = '/home/ma-user/work/code_dev/siming'

def show_sample_num(src_dir):
    sample_list = list(os.listdir(src_dir))
    num = sum([1 for sample in sample_list if os.path.isfile(os.path.join(src_dir, sample))])
    return num


if __name__ == '__main__':
    src_dir = f'{ROOT_PATH}/datasets/iNaturalist/images'

    sample_num = show_sample_num(src_dir)
    logger.debug(f'sample_num: {sample_num}')