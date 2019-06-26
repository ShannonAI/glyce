#!/usr/bin/env python
"""
# encoding: utf-8
# @author  : NiePing
# @contact : ping.nie@pku.edu.cn
# @versoin : 1.0
# @license: Apache Licence
# @file    : run_sei.py
# @time    : 2019-01-19 15:42
"""

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path) 


import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


import copy
import argparse
import datetime
import logging
from sklearn.metrics import f1_score, classification_report, accuracy_score


from glyce.models.sem_sim.model import BIMPM
from glyce.dataset_readers.semantic_similar_data import *



def test(model, args, data_loader, logger):

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0
    nbatch = 0
    label_all = []
    pred_all = []
    with torch.no_grad():
        glyph_lossp_total = 0.
        glyph_lossh_total = 0.
        for idx, minibatch in enumerate(data_loader):
            nbatch += 1
            s1, s2 = 'q1', 'q2'
            s1, s2 = getattr(minibatch, s1), getattr(minibatch, s2)

            if args.bgtt >= 0:
                if s1.size()[1] > args.bgtt:
                    s1 = s1[:, :args.bgtt]
                if s2.size()[1] > args.bgtt:
                    s2 = s2[:, :args.bgtt]
            kwargs = {'p': s1, 'h': s2}
            pred, glyph_lossp, glyph_lossh = model(**kwargs)
            glyph_lossp_total += glyph_lossp if args.use_glyph_emb else glyph_lossp
            glyph_lossh_total += glyph_lossh if args.use_glyph_emb else glyph_lossh

            batch_loss = criterion(pred, minibatch.label)
            loss += batch_loss.item()

            _, pred = pred.max(dim=1)
            pred_all += [x.item() for x in pred]
            label_all += [x.item() for x in minibatch.label]
            acc += (pred == minibatch.label).sum().float()
            size += len(pred)
    print("glyph_lossp:", glyph_lossp_total / nbatch, "glyph_lossh:", glyph_lossh_total / nbatch)
    acc /= size
    acc = acc.cpu().item()
    return loss / nbatch, acc, pred_all, label_all

def train(args, logger):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data_type == 'BQ':
        train_loader, valid_loader, test_loader, dictionary, bq_data = get_data_loader(args)
    else:
        raise NotImplementedError("only BQ data is possible")
    setattr(args, 'idx2char', dictionary.itos)
    setattr(args, 'word_vocab_size', len(dictionary.itos))
    if args.pretrain:
        model = torch.load(args.pretrain)
        for m in model.glyph_embedder.modules():
            if isinstance(m, torch.nn.Conv2d):
                for param in m.parameters():   # 停止训练cnn
                    param.requires_grad = False
    else:
        model = BIMPM(args)
    if args.gpu_id > -1:
        model.cuda()
    logger.info(model)

    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.1, verbose=True, min_lr=1e-8, threshold=1e-4)
    criterion = nn.CrossEntropyLoss()
    logger.info('start training...')
    best_epoch = 0
    max_dev_acc, max_test_acc = 0, 0

    loss, last_epoch = 0, -1
    model.train()

    for epoch in range(args.epochs):
        present_epoch = epoch
        nbatch = 0
        glyph_lossp_total = 0.
        glyph_lossh_total = 0.
        for idx, minibatch in enumerate(train_loader):
            nbatch += 1
            s1, s2 = 'q1', 'q2'
            s1, s2 = getattr(minibatch, s1), getattr(minibatch, s2)

            if args.bgtt >= 0:
                if s1.size()[1] > args.bgtt:
                    s1 = s1[:, :args.bgtt]
                if s2.size()[1] > args.bgtt:
                    s2 = s2[:, :args.bgtt]
            kwargs = {'p': s1, 'h': s2}

            pred, glyph_lossp, glyph_lossh = model(**kwargs)
            glyph_lossp_total += glyph_lossp if args.use_glyph_emb else glyph_lossp
            glyph_lossh_total += glyph_lossh if args.use_glyph_emb else glyph_lossh
            optimizer.zero_grad()
            batch_loss = criterion(pred, minibatch.label)
            loss_sei = batch_loss * (1 - args.glyph_ratio) + args.glyph_ratio * args.glyph_decay**(epoch+1) * (glyph_lossp + glyph_lossh) / 2
            loss_sei.backward()
            loss += batch_loss.item()
            optimizer.step()


        if  present_epoch > last_epoch:
            logger.info(f'epcoh: {present_epoch + 1}')
            loss /= nbatch
            dev_loss, dev_acc, dev_pred_all, dev_label_all = test(model, args, valid_loader, logger)
            test_loss, test_acc, test_pred_all, test_label_all = test(model, args, test_loader, logger)
            logger.info("test report:")

            last_epoch = present_epoch
            scheduler.step(dev_loss)
            print("glyph_lossp:", glyph_lossp_total / nbatch, "glyph_lossh:", glyph_lossh_total / nbatch)
            logger.info(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f} / test loss: {test_loss:.3f}'
                  f' / dev acc: {dev_acc:.3f} / test acc: {test_acc:.3f} / from {nbatch} batchs average')

            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)
                best_epoch = present_epoch
            loss = 0
            model.train()

        if present_epoch > best_epoch + args.early_stopping_threshold:
            logger.info("early stop!!")
            break
    logger.info(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')
    return best_model, best_epoch, max_dev_acc, max_test_acc


def one_trian():
    parser = argparse.ArgumentParser(description='PyTorch Bimpm sei Model')
    parser.add_argument('--data', type=str,
                        default='/data/nfsdata/nlp/datasets/sentence_pair/bq_corpus_torch10/',
                        help='location of the data corpus')
    parser.add_argument('--pretrain', type=str, default='', help='pretrained_file')
    parser.add_argument('--save', type=str, default='/data/nfsdata/nlp/datasets/sentence_pair/bq_corpus_torch10/', help='path to save the final model')
    parser.add_argument('--epochs', type=int, default=500, help='upper epoch limit')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--early_stopping_threshold', type=int, default=10,
                        help='number of epoches to stop if validation ppl do not decrease')
    parser.add_argument('--reload', action='store_true', help='reload data from files or load from cache(default)')
    # parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')

    parser.add_argument('--gpu_id', type=int, default=0, help='the gpu id to train language model')
    parser.add_argument('--data_type', default='BQ', help='available: BQ')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--num_perspective', default=20, type=int)

    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--bgtt', default=50, type=int,
                        help='max length of input sentences model can accept, if -1, it accepts any length')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--use_glyph_emb', default=False, action='store_true')
    parser.add_argument('--word_emb_size', default=300, type=int)
    parser.add_argument('--glyph_emb_size', default=64, type=int)
    parser.add_argument('--glyph_out_size', default=300, type=int)
    parser.add_argument('--use_highway', default=True, action='store_true')
    parser.add_argument('--cnn_dropout', default=0.5, type=float)
    parser.add_argument('--glyph_decay', default=1.0, type=float)
    parser.add_argument('--glyph_ratio', default=1.0, type=float)

    args = parser.parse_args()
    setattr(args, 'class_size', 2)
    setattr(args, 'model_time', datetime.datetime.now().strftime('%H:%M:%S'))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    torch.manual_seed(args.seed)

    save_path = os.path.join(args.save, "BIMPM" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(save_path, 'run.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    print("start training...")
    best_model, best_epoch, max_dev_acc, max_test_acc= train(args, logger)



    torch.save(best_model, os.path.join(save_path, 'model.pkl'))
    print("train finished!")

def main():
    pass


if __name__ == '__main__':
    one_trian()








