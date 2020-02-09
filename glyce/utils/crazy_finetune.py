# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: crazy_finetune.py
@time: 19-1-2 下午9:50

写for循环疯狂调参
python main.py --highway  --nfeat 128 --use_wubi --gpu_id 3
"""

import os 
import sys


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import logging
from itertools import product


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# font_name = '/data/nfsdata/nlp/fonts/useful'
font_name = os.path.join(root_path, "fonts")

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('run.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


# list里的第一个元素是默认设置
finetune_options = {
    'word_embsize': [2048],
    'num_fonts_concat': [0],
    'output_size': [2048],
    'gpu_id': [2],
}


def construct_command(setting):
    command = 'python -m glyph_embedding.experiments.run_lm'
    for feature, option in setting.items():
        if option is True:
            command += F' --{feature}'
        elif option is False:
            command += ''
        else:
            command += F' --{feature} {option}'
    return command


def traverse():
    """以默认配置为基准，每次只调一个参数，m个参数，每个参数n个选项，总共运行m*(n-1)次"""
    default_setting = {k: v[0] for k, v in finetune_options.items()}
    for feature in finetune_options:
        for i, option in enumerate(finetune_options[feature]):
            if i and default_setting[feature] != option:  # 默认设置
                setting = default_setting
                setting[feature] = option
                command = construct_command(setting)
                logger.info(command)
                try:
                    message = os.popen(command).read()
                except:
                    message = '进程启动失败!!'
                logger.info(message)


def grid_search():
    """以grid search的方式调参"""
    for vs in product(*finetune_options.values()):
        setting = {}
        for k, v in zip(finetune_options.keys(), vs):
            setting[k] = v
        command = construct_command(setting)
        logger.info(command)
        try:
            message = os.popen(command).read()
        except:
            message = '进程启动失败!!'
        logger.info(message)


if __name__ == '__main__':
    grid_search()
