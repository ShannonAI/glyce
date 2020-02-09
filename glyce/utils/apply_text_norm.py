#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last udpate: 2019.03.29 
# First create: 2019.03.29 
# Description:
# text_norm.py
# ---------------------------------------------------
# bert -> use chinese punction symbol 


import os 
import sys 

root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

    
import re 
import json 
import numpy as np 
import _pickle as pkl 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)



def chinese_to_english_punct(sent, dims=1, replace_lst=["，", "。", "！", "？", "；", "（", "）", "＠", "＃", "【", "】", "+", "=", "-", "：", "“",  "”",  "‘",  "’",  "》",  "《",  "「",  "」"], target_lst =  [",", ".", "!", "?", ";", "(", ")", "@", "#", "[", "]", "+", "=", "-", ":", '"', '"', "'", "'", ">", "<", "{", "}", ]):
    # 中国，中文，标点符号！你好？１２３４５＠＃【】+=-（）
    if dims == 1:
        for item_idx, (replace_item, target_item) in enumerate(zip(replace_lst, target_lst)):
            if replace_item not in sent:
                continue 
            sent = sent.replace(replace_item, target_item)
        return sent 
    elif dims == 2:
        tar_lst = []
        for sent_item in sent:
            tmp_sent = chinese_to_english_punct(sent_item, dims=1)
            tar_lst.append(tmp_sent)
        return tar_lst 


def full2half(sent, dims=1):
    if dims == 1:
        str_char = ""
        for char in sent:
            num = ord(char)
            if num == 0x3000:
                num = 32 
            elif 0xFF01 <= num <= 0xFF5E:
                num -= 0xfee0 
            num = chr(num)
            str_char += num 
        return str_char 
    elif dims == 2:
        str_chars = []
        for s_item in s:
            tmp_chars = full2half(s_item, dims=1)
            str_chars.append(tmp_chars)
        return str_chars 



def process_sent(sent, dims=1):
    # replace chinese punction into english punction 
    sent = chinese_to_english_punct(sent, dims=dims)
    # full2half replace 
    sent = full2half(sent, dims=dims)
    return sent 


if __name__ == "__main__":
    sent = "我在天安门广场。"
    print(sent)
    sent = process_sent(sent, dims=1)
    print(sent)
