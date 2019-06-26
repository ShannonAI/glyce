#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.04.04 
# First create: 2019.04.04 
# length statisic 


import os 
import sys 

root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import json 
import numpy as np 



def read_lines(data_repo):
    with open(data_repo, "r") as f:
        data_lines = f.readlines()
    return data_lines 



def collect_sentence(data_repo, datatype_sign):
    if datatype_sign == "train":
        data_path = os.path.join(data_repo, "train.char.bmes")
    elif datatype_sign == "dev":
        data_path = os.path.join(data_repo, "dev.char.bmes")
    elif datatype_sign == "test":
        data_path = os.path.join(data_repo, "test.char.bmes")
    else:
        raise ValueError 

    data_lines = read_lines(data_path)

    sent_lst = []

    for data_item in data_lines:
        data_item = data_item.split("\t")
        text_item = data_item[0].split(" ")
        sent_lst.append(text_item)

    return sent_lst 


def length_statistic(sent_lst):
    length_lst = [len(tmp) for tmp in sent_lst]

    print("total number of lengths :") 
    print(len(length_lst))
    print("total number of length larger than 100 :") 
    print(len([tmp for tmp in length_lst if tmp > 100]))
    print("total number of length larger than 50 :")
    print(len([tmp for tmp in length_lst if tmp > 50])) 
    print("total number of length larger than 70 :")
    print(len([tmp for tmp in length_lst if tmp > 70]))
    print("Mean length of sents : ")
    print(round(sum(length_lst) / len(length_lst), 4))
    print("Max length of sents : ")
    print(max(length_lst))
    print("Min length of sents : ")
    print(min(length_lst))
    


def main(data_repo, datatype_sign):
    sent_lst = collect_sentence(data_repo, datatype_sign)
    length_statistic(sent_lst)



if __name__ == "__main__":
    data_repo = "/data/ResumeNER"
    # datatype_sign = "train"
    datatype_sign = "test"
    main(data_repo, datatype_sign)
