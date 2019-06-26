#!/usr/bin/env python3
# -*- coding: utf-8 -*- 




# Author: Xiaoy LI
# Last update: 2019.03.23 
# First create: 2019.03.23 
# Description:
# run_tagger.py 


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import csv 
import logging 
import argparse 
import numpy as np 
from tqdm import tqdm 


import torch 
import torch.nn as nn 
from torch.optim import Adam 
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler 


from glyce.utils.tokenization import BertTokenizer  
from glyce.utils.optimization import BertAdam, warmup_linear 
from glyce.utils.file_utils import PYTORCH_PRETRAINED_BERT_CACHE 
from glyce.dataset_readers.bert_config import Config 
from glyce.dataset_readers.bert_ner import * 
from glyce.dataset_readers.bert_pos import * 
from glyce.dataset_readers.bert_cws import * 
from glyce.models.bert.bert_tagger import BertTagger 
from glyce.dataset_readers.bert_data_utils import convert_examples_to_features  
from glyce.utils.metrics.tagging_evaluate_funcs import compute_performance 


logging.basicConfig()
logger = logging.getLogger(__name__)




def args_parser():
    # start parser 
    parser = argparse.ArgumentParser()


    # required parameters 
    parser.add_argument("--config_path", default="/home/lixiaoya/dataset/", type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")
    parser.add_argument("--bert_model", default=None, type=str, required=True, 
        help="bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--task_name", default=None, type=str)
    # # other parameters 
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--max_seq_length", default=128, 
        type=int, help="the maximum total input sequence length after ")
    parser.add_argument("--do_train", action="store_true", 
        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", 
        help="set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int) 
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3306)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_dir", type=str, default="/home/lixiaoya/bert_output")
    parser.add_argument("--data_sign", type=str, default="msra_ner")
    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps 

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError 
    os.makedirs(args.output_dir, exist_ok=True)

    return args 


def load_data(config):
    # load some data and processor 
    # data_processor = MsraNerProcessor()
    if config.data_sign == "msra_ner":
        data_processor = MsraNERProcessor()
    elif config.data_sign == "resume_ner":
        data_processor = ResumeNERProcessor()
    elif config.data_sign == "ontonotes_ner":
        data_processor = OntoNotesNERProcessor()
    elif config.data_sign == "ctb5_pos":
        data_processor = Ctb5POSProcessor()
    elif config.data_sign == "ctb6_pos":
        data_processor = Ctb6POSProcessor()
    elif config.data_sign == "ud1_pos":
        data_processor = Ud1POSProcessor()
    elif config.data_sign == "ctb6_cws":
        data_processor = Ctb6CWSProcessor() 
    elif config.data_sign == "pku_cws":
        data_processor = PkuCWSProcessor() 
    elif config.data_sign == "msr_cws":
        data_processor = MsrCWSProcessor() 
    else:
        raise ValueError 


    label_list = data_processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    # load data exampels 
    train_examples = data_processor.get_train_examples(config.data_dir)
    dev_examples = data_processor.get_dev_examples(config.data_dir)
    test_examples = data_processor.get_test_examples(config.data_dir)

    # convert data example into featrues 
    train_features = convert_examples_to_features(train_examples, label_list, config.max_seq_length, tokenizer)
    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)
    # train_sampler = DistributedSampler(train_data)
    train_sampler = RandomSampler(train_data)

    dev_features = convert_examples_to_features(dev_examples, label_list, config.max_seq_length, tokenizer)
    dev_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    dev_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    dev_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    dev_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
    dev_data = TensorDataset(dev_input_ids, dev_input_mask, dev_segment_ids, dev_label_ids)
    # dev_sampler = DistributedSampler(dev_data)
    dev_sampler = RandomSampler(dev_data)

    test_features = convert_examples_to_features(test_examples, label_list, config.max_seq_length, tokenizer)
    test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
    # test_sampler = DistributedSampler(test_data)
    test_sampler = RandomSampler(test_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, \
        batch_size=config.train_batch_size)

    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, \
        batch_size=config.dev_batch_size)

    test_dataloader = DataLoader(test_data, sampler=test_sampler, \
        batch_size=config.test_batch_size)

    num_train_steps = int(len(train_examples) / config.train_batch_size * config.num_train_epochs) 
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list 


def load_model(config, num_train_steps, label_list):
    # device = torch.device(torch.cuda.is_available())
    device = torch.device("cuda") 
    n_gpu = torch.cuda.device_count()
    model = BertTagger(config, num_labels=len(label_list)) 
    # model = BertForTagger.from_pretrained(config.bert_model, num_labels=13)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare  optimzier 
    param_optimizer = list(model.named_parameters())

        
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    # optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate) 
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup_proportion, t_total=num_train_steps, max_grad_norm=config.clip_grad) 

    return model, optimizer, device, n_gpu


def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader, config, \
    device, n_gpu, label_list):
    global_step = 0 
    nb_tr_steps = 0 
    tr_loss = 0 

    dev_best_acc = 0 
    dev_best_precision = 0 
    dev_best_recall = 0 
    dev_best_f1 = 0 
    dev_best_loss = 10000000000000


    test_best_acc = 0 
    test_best_precision = 0 
    test_best_recall = 0 
    test_best_f1 = 0 
    test_best_loss = 1000000000000000

    model.train()

    for idx in range(int(config.num_train_epochs)):
        tr_loss = 0 
        nb_tr_examples, nb_tr_steps = 0, 0 
        print("#######"*10)
        print("EPOCH: ", str(idx))
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch 
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()

            model.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad) 

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1 

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                # optimizer.zero_grad()
                global_step += 1 

            if nb_tr_steps % config.checkpoint == 0:
                print("-*-"*15)
                print("current training loss is : ")
                print(loss.item())
                tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config, device, n_gpu, label_list, eval_sign="dev")
                print("......"*10)
                print("DEV: loss, acc, precision, recall, f1")
                print(tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)

                if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
                    dev_best_acc = tmp_dev_acc 
                    dev_best_loss = tmp_dev_loss 
                    dev_best_precision = tmp_dev_prec 
                    dev_best_recall = tmp_dev_rec 
                    dev_best_f1 = tmp_dev_f1 

                    tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model, test_dataloader, config, device, n_gpu, label_list, eval_sign="test")
                    print("......"*10)
                    print("TEST: loss, acc, precision, recall, f1")
                    print(tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1)

                    if tmp_test_f1 > test_best_f1 or tmp_test_acc > test_best_acc:
                        test_best_acc = tmp_test_acc 
                        test_best_loss = tmp_test_loss 
                        test_best_precision = tmp_test_prec 
                        test_best_recall = tmp_test_rec 
                        test_best_f1 = tmp_test_f1 

                        # export model 
                        if config.export_model:
                            model_to_save = model.module if hasattr(model, "module") else model 
                            output_model_file = os.path.join(config.output_dir, "bert_finetune_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)

                print("-*-"*15)

    # export a trained mdoel 
    model_to_save = model 
    output_model_file = os.path.join(config.output_dir, "bert_model.bin")
    if config.export_model == "True":
        torch.save(model_to_save.state_dict(), output_model_file)


    print("=&="*15)
    print("DEV: current best precision, recall, f1, acc, loss ")
    print(dev_best_precision, dev_best_recall, dev_best_f1, dev_best_acc, dev_best_loss)
    print("TEST: current best precision, recall, f1, acc, loss ")
    print(test_best_precision, test_best_recall, test_best_f1, test_best_acc, test_best_loss)
    print("=&="*15)

def eval_checkpoint(model_object, eval_dataloader, config, \
    device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader 
    model_object.eval()

    idx2label = {i: label for i, label in enumerate(label_list)}

    eval_loss = 0 
    pred_lst = []
    mask_lst = []
    gold_lst = []
    eval_steps = 0 

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, label_ids)
            logits = model_object(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        # logits = np.argmax(logits, axis=-1) 
        label_ids = label_ids.to("cpu").numpy()
        input_mask = input_mask.to("cpu").numpy() 
        reshape_lst = label_ids.shape
        logits = np.reshape(logits, (reshape_lst[0], reshape_lst[1], -1))
        logits = np.argmax(logits, axis=-1) 

        logits = logits.tolist()
        input_mask = input_mask.tolist()         
        label_ids = label_ids.tolist()

        eval_loss += tmp_eval_loss.mean().item()

        pred_lst += logits 
        gold_lst += label_ids 
        mask_lst += input_mask 
        eval_steps += 1   
     
    eval_accuracy, eval_precision, eval_recall, eval_f1 = compute_performance(pred_lst, gold_lst, mask_lst, label_list, dims=2)  

    average_loss = round(eval_loss / eval_steps, 4)  
    eval_f1 = round(eval_f1 , 4)
    eval_precision = round(eval_precision , 4)
    eval_recall = round(eval_recall , 4) 
    eval_accuracy = round(eval_accuracy , 4) 

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 



def merge_config(args_config):
    model_config_path = args_config.config_path 
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def main():
    args_config = args_parser() 
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, device, n_gpu = load_model(config, num_train_steps, label_list)
    train(model, optimizer, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)
    



if __name__ == "__main__":
    main()
