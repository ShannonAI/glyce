# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-07-06 11:08:27


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import argparse
import datetime
import gc
import logging
import random
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim


from glyce.models.latticeLSTM.bilstmcrf import BiLSTMCRF as SeqModel
from glyce.models.latticeLSTM.utils.data import Data
from glyce.models.latticeLSTM.utils.metric import get_ner_fmeasure


parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CRF')
parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
parser.add_argument('--name', type=str, default='CTB9POS')
parser.add_argument('--mode', type=str, default='char')
parser.add_argument('--data_dir', type=str, default='/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/')
parser.add_argument('--raw', type=str)
parser.add_argument('--loadmodel', type=str)
parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--gaz_dropout', type=float, default=0.5)
parser.add_argument('--HP_lr', type=float, default=0.01)
parser.add_argument('--HP_dropout', type=float, default=0.5)
parser.add_argument('--HP_use_glyph', action='store_true')
parser.add_argument('--HP_glyph_ratio', type=float, default=0.1)
parser.add_argument('--HP_font_channels', type=int, default=2)
parser.add_argument('--HP_glyph_highway', action='store_true')
parser.add_argument('--HP_glyph_layernorm', action='store_true')
parser.add_argument('--HP_glyph_batchnorm', action='store_true')
parser.add_argument('--HP_glyph_embsize', type=int, default=64)
parser.add_argument('--HP_glyph_output_size', type=int, default=64)
parser.add_argument('--HP_glyph_dropout', type=float, default=0.7)
parser.add_argument('--HP_glyph_cnn_dropout', type=float, default=0.5)
parser.add_argument('--setting_str', type=str, default='')
parser.add_argument('--src_folder', type=str, default='/data/nfsdata/nlp/projects/wuwei')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
save_dir = F'{args.src_folder}/{args.name}.'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
logger = logging.getLogger()  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(save_dir, 'run.log'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_gaz_file(gaz_file)
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)
    data.fix_alphabet()
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        # logger.info "p:",pred, pred_tag.tolist()
        # logger.info "g:", gold, gold_tag.tolist()
        assert(len(pred)==len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def load_data_setting(save_dir):
    with open(save_dir, 'rb') as fp:
        data = torch.load(fp)
    logger.info("Data setting loaded from file: " + save_dir)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1-decay_rate)**epoch)
    logger.info(F" Learning rate is setted as: {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        logger.info("Error: wrong evaluate name," + name)
    pred_results = []
    gold_results = []
    model.eval()
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//data.HP_batch_size+1

    for batch_id in range(total_batch):
        start = batch_id*data.HP_batch_size
        end = (batch_id+1)*data.HP_batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(gaz_list,batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # logger.info("tag_seq", tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results  


def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    """
        input: list of words, chars and labels, various length. [[words,biwords,chars,gaz, labels],[words,biwords,chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len) 
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order 
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len) 
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile =  volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)),volatile =  volatile_flag).byte()
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    ## not reorder label
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(list(map(max, length_list)))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile =  volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # logger.info len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    
    #  keep the gaz_list in orignial order
    
    gaz_list = [ gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def train(data, save_model_dir, seg=True):
    logger.info("Training model...")
    data.show_data_summary()
    model = SeqModel(data)
    logger.info("finished built model.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    best_dev = -1
    data.HP_iteration = 100
    #  start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        logger.info(("Epoch: %s/%s" %(idx,data.HP_iteration)))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        train_num = len(data.train_Ids)
        total_batch = train_num//data.HP_batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*data.HP_batch_size
            end = min((batch_id+1)*data.HP_batch_size, train_num)
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            gaz_list,  batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data[0]
            total_loss += loss.data[0]
            batch_loss += loss

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                logger.info(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token)))
                sys.stdout.flush()
                sample_loss = 0
            if end % data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        logger.info(("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token)))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        logger.info(("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (idx, epoch_cost, train_num/epoch_cost, total_loss)))
        speed, acc, p, r, f, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if seg:
            current_score = f
            logger.info(("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f)))
        else:
            current_score = acc
            logger.info(("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc)))

        if current_score > best_dev:
            if seg:
                logger.info(F"Exceed previous best f score: {best_dev}")
            else:
                logger.info(F"Exceed previous best acc score: {best_dev}")
            model_name = os.path.join(save_model_dir, 'saved.model')
            torch.save(model.state_dict(), model_name)
            best_dev = current_score 
        # ## decode test
        # speed, acc, p, r, f, _ = evaluate(data, model, "test")
        # test_finish = time.time()
        # test_cost = test_finish - dev_finish
        # if seg:
        #     logger.info(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f)))
        # else:
        #     logger.info(("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc)))
        gc.collect() 


def load_model_decode(save_dir, data):
    logger.info("Load Model from file: " + save_dir)
    model = SeqModel(data)
    model.load_state_dict(torch.load(save_dir))
    logger.info(F"Decode dev data ...")
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, 'dev')
    end_time = time.time()
    time_cost = end_time - start_time
    logger.info(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%('dev', time_cost, speed, acc, p, r, f)))

    logger.info(F"Decode test data ...")
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, 'test')
    end_time = time.time()
    time_cost = end_time - start_time
    logger.info(("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%('test', time_cost, speed, acc, p, r, f)))


if __name__ == '__main__':
    char_emb = '/data/nfsdata/nlp/embeddings/chinese/gigaword/gigaword_chn.all.a2b.uni.ite50.vec'
    bichar_emb = ''
    # bichar_emb = '/data/nfsdata/nlp/embeddings/chinese/gigaword/gigaword_chn.all.a2b.bi.ite50.vec'
    ctb_gaz = '/data/nfsdata/nlp/embeddings/chinese/ctb/ctb.50d.vec'  # NER
    wiki_gaz = '/data/nfsdata/nlp/embeddings/chinese/wiki/zh.wiki.bpe.vs200000.d50.w2v.txt'
    gaz_file = ctb_gaz if 'NER' in args.name else wiki_gaz

    train_file = F'{args.data_dir}/{args.name}/train.{args.mode}.bmes'
    dev_file = F'{args.data_dir}/{args.name}/dev.{args.mode}.bmes'
    test_file = F'{args.data_dir}/{args.name}/test.{args.mode}.bmes'
    
    logger.info("Train file:" + train_file)
    logger.info("Dev file:" + dev_file)
    logger.info("Test file:" + test_file)
    logger.info("Char emb:" + char_emb)
    logger.info("Bichar emb:" + bichar_emb)
    logger.info("Gaz file:" + gaz_file)
    logger.info("Save dir:" + save_dir)
    sys.stdout.flush()
    
    if args.status == 'train':
        data = Data()
        data.HP_use_char = False
        data.use_bigram = False if 'NER' in args.name else True  # ner: False, cws: True
        data.gaz_dropout = args.gaz_dropout
        data.HP_lr = 0.015 if 'NER' in args.name else 0.01
        data.HP_dropout = args.HP_dropout
        data.HP_use_glyph = args.HP_use_glyph
        data.HP_glyph_ratio = args.HP_glyph_ratio
        data.HP_font_channels = args.HP_font_channels
        data.HP_glyph_highway = args.HP_glyph_highway
        data.HP_glyph_embsize = args.HP_glyph_embsize
        data.HP_glyph_output_size = args.HP_glyph_output_size
        data.HP_glyph_dropout = args.HP_glyph_dropout
        data.HP_glyph_cnn_dropout = args.HP_glyph_cnn_dropout
        data.HP_glyph_batchnorm = args.HP_glyph_batchnorm
        data.HP_glyph_layernorm = args.HP_glyph_layernorm
        data.norm_gaz_emb = False if 'NER' in args.name else True  # ner: False, cws: True

        data.HP_fix_gaz_emb = False
        data_initialization(data, gaz_file, train_file, dev_file, test_file)
        data.generate_instance_with_gaz(train_file, 'train')
        data.generate_instance_with_gaz(dev_file, 'dev')
        data.generate_instance_with_gaz(test_file, 'test')
        data.build_word_pretrain_emb(char_emb)
        data.build_biword_pretrain_emb(bichar_emb)
        data.build_gaz_pretrain_emb(gaz_file)
        torch.save(data, save_dir + '/data.set')
        data = torch.load(save_dir + '/data.set')
        train(data, save_dir)
    elif args.status == 'test':
        data = load_data_setting(args.loadmodel + '/data.set')
        load_model_decode(args.loadmodel + '/saved.model', data)
        # load_model_decode(args.loadmodel + '/saved.model', data, 'test')
    # elif args.status == 'decode':
    #     data = load_data_setting(args.loadmodel + '/data.set')
    #     data.generate_instance_with_gaz(args.raw, 'raw')
    #     decode_results = load_model_decode(args.loadmodel + '/saved.model', data, 'raw')
    #     data.write_decoded_results(args.loadmodel + '/decoded.output', decode_results, 'raw')
    else:
        logger.info("Invalid argument! Please use valid arguments! (train/test/decode)")
