#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


# Author: Xiaoy LI 
# Last udpate: 2019.02.18 
# First create: 2019.02.18 
# Description:
# change the repo path for running the process 


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 


import numpy as np 


import glyce.models.srl.inter_utils as inter_utils 
from glyce.models.srl.data_utils import _PAD_, _UNK_
from glyce.models.srl.utils import get_torch_variable_from_np, get_data


def pred_acc(target, predict, pred2idx):
    num_correct = 0
    batch_total = 0
    correct_pred = 0
    pred_total = 0
    for i in range(len(target)):
        pred_i = predict[i]
        golden_i = target[i]
        if golden_i == pred2idx[_PAD_]:
            continue
        batch_total += 1
        if pred_i == pred2idx[_UNK_]:
            pred_i = pred2idx['_']
        if golden_i == pred2idx[_UNK_]:
            golden_i = pred2idx['_']
        if pred_i == golden_i:
            num_correct += 1
        if golden_i != pred2idx['_']:
            pred_total += 1
        if golden_i != pred2idx['_'] and pred_i == golden_i:
            correct_pred += 1

    print('accurate:{:.2f} pred accuracy:{:.2f}'.format(num_correct/batch_total*100, correct_pred/pred_total*100))

    return (num_correct/batch_total, correct_pred/pred_total)


def pred_recog_score(target, predict, pred2idx):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    num_correct = 0
    batch_total = 0
    correct_pred = 0
    pred_total = 0
    for i in range(len(target)):
        pred_i = predict[i]
        golden_i = target[i]
        if golden_i == pred2idx[_PAD_]:
            continue
        batch_total += 1
        if pred_i == pred2idx[_UNK_]:
            pred_i = pred2idx['_']
        if golden_i == pred2idx[_UNK_]:
            golden_i = pred2idx['_']
        if pred_i == golden_i:
            num_correct += 1
        if golden_i != pred2idx['_']:
            pred_total += 1
        if golden_i != pred2idx['_'] and pred_i == golden_i:
            correct_pred += 1
            tp += 1
        if golden_i == pred2idx['_'] and pred_i == golden_i:
            tn += 1
        if golden_i != pred2idx['_'] and pred_i != golden_i:
            fp += 1
        if golden_i == pred2idx['_'] and pred_i != golden_i:
            fn += 1

    p = tp / (tp + fp + 1e-13)

    r = tp / (tp + fn + 1e-13)

    f1 = 2 * p * r / (p + r + 1e-13)

    print('accurate:{:.2f} pred accuracy:{:.2f} P:{:.2f} R:{:.2f} F1:{:.2f}'.format(num_correct/batch_total*100, correct_pred/pred_total*100, p*100, r*100, f1*100))

    return (num_correct/batch_total, correct_pred/pred_total)


def sem_f1_score(target, predict, argument2idx, unify_pred = False, predicate_correct=0, predicate_sum=0):
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    total = 0

    if unify_pred:
        predicate_correct = 0
        predicate_sum = 0

    for i in range(len(target)):
        for j in range(len(target[i])):
            pred_i = predict[i][j]
            golden_i = target[i][j]

            if unify_pred and j == 0:
                predicate_sum += 1
                if pred_i == golden_i:
                    predicate_correct += 1
            else:
                if golden_i == argument2idx[_PAD_]:
                    continue
                total += 1
                if pred_i == argument2idx[_UNK_]:
                    pred_i = argument2idx['_']
                if golden_i == argument2idx[_UNK_]:
                    golden_i = argument2idx['_']
                if pred_i != argument2idx['_']:
                    predict_args += 1
                if golden_i != argument2idx['_']:
                    golden_args += 1
                if golden_i != argument2idx['_'] and pred_i == golden_i:
                    correct_args += 1
                if pred_i == golden_i:
                    num_correct += 1

    P = (correct_args + predicate_correct) / (predict_args + predicate_sum + 1e-13)

    R = (correct_args + predicate_correct) / (golden_args + predicate_sum + 1e-13)

    NP = correct_args / (predict_args + 1e-13)

    NR = correct_args / (golden_args + 1e-13)
        
    F1 = 2 * P * R / (P + R + 1e-13)

    NF1 = 2 * NP * NR / (NP + NR + 1e-13)

    print('\teval accurate:{:.2f} predict:{} golden:{} correct:{} P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(num_correct/total*100, predict_args, golden_args, correct_args, P*100, R*100, F1*100, NP*100, NR *100, NF1 * 100))

    return (P, R, F1, NP, NR, NF1)

def pruning_sem_f1_score(target, predict, out_of_pruning, argument2idx, unify_pred = False, predicate_correct=0, predicate_sum=0):
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    total = 0

    if unify_pred:
        predicate_correct = 0
        predicate_sum = 0

    for i in range(len(target)):
        for j in range(len(target[i])):
            pred_i = predict[i][j]
            golden_i = target[i][j]

            if unify_pred and j == 0:
                predicate_sum += 1
                if pred_i == golden_i:
                    predicate_correct += 1
            else:
                if golden_i == argument2idx[_PAD_]:
                    continue
                total += 1
                if pred_i == argument2idx[_UNK_]:
                    pred_i = argument2idx['_']
                if golden_i == argument2idx[_UNK_]:
                    golden_i = argument2idx['_']
                if pred_i != argument2idx['_']:
                    predict_args += 1
                if golden_i != argument2idx['_']:
                    golden_args += 1
                if golden_i != argument2idx['_'] and pred_i == golden_i:
                    correct_args += 1
                if pred_i == golden_i:
                    num_correct += 1

    P = (correct_args + predicate_correct) / (predict_args + predicate_sum + 1e-13)

    R = (correct_args + predicate_correct) / (golden_args + out_of_pruning + predicate_sum + 1e-13)
        
    NP = correct_args / (predict_args + 1e-13)

    NR = correct_args / (golden_args + 1e-13)
        
    F1 = 2 * P * R / (P + R + 1e-13)

    NF1 = 2 * NP * NR / (NP + NR + 1e-13)

    print('\tpred acc:{:.2f} arg acc:{:.2f} predict:{} golden:{} correct:{} P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(predicate_correct/predicate_sum*100, num_correct/total*100, predict_args, golden_args + out_of_pruning, correct_args, P*100, R*100, F1*100, NP*100, NR *100, NF1 * 100))

    return (P, R, F1, NP, NR, NF1)

def eval_train_batch(epoch,batch_i,loss,golden_batch,predict_batch,argument2idx):
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    batch_total = 0
    for i in range(len(golden_batch)):
        pred_i = predict_batch[i]
        golden_i = golden_batch[i]
        if golden_i == argument2idx[_PAD_]:
            continue
        batch_total += 1
        if pred_i == argument2idx[_UNK_]:
            pred_i = argument2idx['_']
        if golden_i == argument2idx[_UNK_]:
            golden_i = argument2idx['_']
        if pred_i != argument2idx['_']:
            predict_args += 1
        if golden_i != argument2idx['_']:
            golden_args += 1
        if golden_i != argument2idx['_'] and pred_i == golden_i:
            correct_args += 1
        if pred_i == golden_i:
            num_correct += 1

    print('epoch {} batch {} loss:{:4f} accurate:{:.2f} predict:{} golden:{} correct:{}'.format(epoch, batch_i, loss, num_correct/batch_total*100, predict_args, golden_args, correct_args))

def eval_train_pred_batch(epoch,batch_i,loss,golden_batch,predict_batch,pred2idx):
    num_correct = 0
    batch_total = 0
    correct_pred = 0
    pred_total = 0
    for i in range(len(golden_batch)):
        pred_i = predict_batch[i]
        golden_i = golden_batch[i]
        if golden_i == pred2idx[_PAD_]:
            continue
        batch_total += 1
        if pred_i == pred2idx[_UNK_]:
            pred_i = pred2idx['_']
        if golden_i == pred2idx[_UNK_]:
            golden_i = pred2idx['_']
        if pred_i == golden_i:
            num_correct += 1
        if golden_i != pred2idx['_']:
            pred_total += 1
        if golden_i != pred2idx['_'] and pred_i == golden_i:
            correct_pred += 1

    print('epoch {} batch {} loss:{:4f} accurate:{:.2f} pred accuracy:{:.2f}'.format(epoch, batch_i, loss, num_correct/batch_total*100, correct_pred/pred_total*100))

def eval_data(model, elmo, dataset, batch_size ,word2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2argument, unify_pred = False, predicate_correct=0, predicate_sum=0):

    model.eval()
    golden = []
    predict = []

    output_data = []
    cur_sentence = None
    cur_sentence_data = None

    for batch_i, input_data in enumerate(inter_utils.get_batch(dataset, batch_size, word2idx,
                                                             lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx)):
        
        target_argument = input_data['argument']
        
        flat_argument = input_data['flat_argument']

        target_batch_variable = get_torch_variable_from_np(flat_argument)

        sentence_id = input_data['sentence_id']
        predicate_id = input_data['predicate_id']
        word_id = input_data['word_id']
        sentence_len =  input_data['sentence_len']
        seq_len = input_data['seq_len']
        bs = input_data['batch_size']
        psl = input_data['pad_seq_len']
        
        out = model(input_data, elmo)

        _, pred = torch.max(out, 1)

        pred = get_data(pred)

        # print(target_argument.shape)
        # print(pred.shape)

        pred = np.reshape(pred, target_argument.shape)

        # golden += flat_argument.tolist()
        # predict += pred

        for idx in range(pred.shape[0]):
            predict.append(list(pred[idx]))
            golden.append(list(target_argument[idx]))

        pre_data = []
        for b in range(len(seq_len)):
            line_data = ['_' for _ in range(sentence_len[b])]
            for s in range(seq_len[b]):
                wid = word_id[b][s]
                line_data[wid-1] = idx2argument[pred[b][s]]
            pre_data.append(line_data)

        for b in range(len(sentence_id)):
            if cur_sentence != sentence_id[b]:
                if cur_sentence_data is not None:
                    output_data.append(cur_sentence_data)
                cur_sentence_data = [[sentence_id[b]]*len(pre_data[b]),pre_data[b]]
                cur_sentence = sentence_id[b]
            else:
                assert cur_sentence_data is not None
                cur_sentence_data.append(pre_data[b])

    if cur_sentence_data is not None and len(cur_sentence_data)>0:
        output_data.append(cur_sentence_data)
    
    score = sem_f1_score(golden, predict, argument2idx, unify_pred, predicate_correct, predicate_sum)

    model.train()

    return score, output_data

def pruning_eval_data(model, elmo, dataset, batch_size, out_of_pruning, word2idx, char2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2argument, unify_pred = False, predicate_correct=0, predicate_sum=0):

    model.eval()
    golden = []
    predict = []

    output_data = []
    cur_sentence = None
    cur_sentence_data = None

    for batch_i, input_data in enumerate(inter_utils.get_batch(dataset, batch_size, word2idx, char2idx,
                                                             lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx)):
        
        target_argument = input_data['argument']

        flat_argument = input_data['flat_argument']

        target_batch_variable = get_torch_variable_from_np(flat_argument)

        sentence_id = input_data['sentence_id']
        predicate_id = input_data['predicate_id']
        word_id = input_data['word_id']
        sentence_len =  input_data['sentence_len']
        seq_len = input_data['seq_len']
        bs = input_data['batch_size']
        psl = input_data['pad_seq_len']
        
        out, _ = model(input_data, elmo)

        # loss = criterion(out, target_batch_variable)

        _, pred = torch.max(out, 1)

        pred = get_data(pred)

        # print(target_argument.shape)
        # print(pred.shape)

        pred = np.reshape(pred, target_argument.shape)

        # golden += flat_argument.tolist()
        # predict += pred

        for idx in range(pred.shape[0]):
            predict.append(list(pred[idx]))
            golden.append(list(target_argument[idx]))

        pre_data = []
        for b in range(len(seq_len)):
            line_data = ['_' for _ in range(sentence_len[b])]
            for s in range(seq_len[b]):
                wid = word_id[b][s]
                line_data[wid-1] = idx2argument[pred[b][s]]
            pre_data.append(line_data)

        for b in range(len(sentence_id)):
            if cur_sentence != sentence_id[b]:
                if cur_sentence_data is not None:
                    output_data.append(cur_sentence_data)
                cur_sentence_data = [[sentence_id[b]]*len(pre_data[b]),pre_data[b]]
                cur_sentence = sentence_id[b]
            else:
                assert cur_sentence_data is not None
                cur_sentence_data.append(pre_data[b])
    
    if cur_sentence_data is not None and len(cur_sentence_data)>0:
        output_data.append(cur_sentence_data)

    score = pruning_sem_f1_score(golden, predict, out_of_pruning, argument2idx, unify_pred, predicate_correct, predicate_sum)

    model.train()

    return score, output_data
"""
def eval_pred_data(model, elmo, dataset, batch_size ,word2idx, lemma2idx, pos2idx, pretrain2idx, pred2idx, idx2pred):
    model.eval()
    golden = []
    predict = []

    output_data = []
    cur_sentence = None
    cur_sentence_data = None

    for batch_i, input_data in enumerate(pred_inter_utils.get_batch(dataset, batch_size, word2idx,
                                                             lemma2idx, pos2idx, pretrain2idx, pred2idx)):
        
        target_pred = input_data['pred']
        
        flat_pred = input_data['flat_pred']

        target_batch_variable = get_torch_variable_from_np(flat_pred)

        sentence_id = input_data['sentence_id']
        word_id = input_data['word_id']
        sentence_len =  input_data['sentence_len']
        seq_len = input_data['seq_len']
        bs = input_data['batch_size']
        psl = input_data['pad_seq_len']
        
        out = model(input_data, elmo)

        _, pred = torch.max(out, 1)

        pred = get_data(pred)

        pred = pred.tolist()

        golden += flat_pred.tolist()
        predict += pred

        pre_data = []
        for b in range(len(seq_len)):
            line_data = ['_' for _ in range(sentence_len[b])]
            for s in range(seq_len[b]):
                wid = word_id[b][s]
                line_data[wid-1] = idx2pred[pred[b * psl + s]]
            pre_data.append(line_data)

        for b in range(len(sentence_id)):
            if cur_sentence != sentence_id[b]:
                if cur_sentence_data is not None:
                    output_data.append(cur_sentence_data)
                cur_sentence_data = [[sentence_id[b]]*len(pre_data[b]),pre_data[b]]
                cur_sentence = sentence_id[b]
            else:
                assert cur_sentence_data is not None
                cur_sentence_data.append(pre_data[b])

        
    if cur_sentence_data is not None and len(cur_sentence_data)>0:
        output_data.append(cur_sentence_data)
    
    score = pred_acc(golden, predict, pred2idx)

    model.train()

    return score, output_data

def eval_pred_recog_data(model, elmo, dataset, batch_size ,word2idx, lemma2idx, pos2idx, pretrain2idx, pred2idx, idx2pred):
    model.eval()
    golden = []
    predict = []

    output_data = []
    cur_sentence = None
    cur_sentence_data = None

    for batch_i, input_data in enumerate(pred_recog_inter_utils.get_batch(dataset, batch_size, word2idx,
                                                             lemma2idx, pos2idx, pretrain2idx, pred2idx)):
        
        target_pred = input_data['pred']
        
        flat_pred = input_data['flat_pred']

        target_batch_variable = get_torch_variable_from_np(flat_pred)

        sentence_id = input_data['sentence_id']
        word_id = input_data['word_id']
        sentence_len =  input_data['sentence_len']
        seq_len = input_data['seq_len']
        bs = input_data['batch_size']
        psl = input_data['pad_seq_len']
        
        out = model(input_data, elmo)

        _, pred = torch.max(out, 1)

        pred = get_data(pred)

        pred = pred.tolist()

        golden += flat_pred.tolist()
        predict += pred

        pre_data = []
        for b in range(len(seq_len)):
            line_data = ['_' for _ in range(sentence_len[b])]
            for s in range(seq_len[b]):
                wid = word_id[b][s]
                line_data[wid-1] = idx2pred[pred[b * psl + s]]
            pre_data.append(line_data)

        for b in range(len(sentence_id)):
            if cur_sentence != sentence_id[b]:
                if cur_sentence_data is not None:
                    output_data.append(cur_sentence_data)
                cur_sentence_data = [[sentence_id[b]]*len(pre_data[b]),pre_data[b]]
                cur_sentence = sentence_id[b]
            else:
                assert cur_sentence_data is not None
                cur_sentence_data.append(pre_data[b])

        
    if cur_sentence_data is not None and len(cur_sentence_data)>0:
        output_data.append(cur_sentence_data)
    
    score = pred_recog_score(golden, predict, pred2idx)

    model.train()

    return score, output_data
"""
