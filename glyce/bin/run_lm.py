# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: run_lm_word
@time: 2019/1/19 12:06

    这一行开始写关于本文件的说明与解释
"""

import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


import json 
import math 
import time 
import argparse
import datetime
import logging


from glyce.dataset_readers.lm_data_word import WordCorpus
from glyce.models.lm.glyph_embedding_for_lm import GlyphEmbeddingForLM


parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model (Word Level)')
parser.add_argument('--data', type=str, default='/data/nfsdata/nlp/datasets/language_modeling/ctb_v6/data/utf8/yuxian_tokens', help='location of the data corpus')
parser.add_argument('--pretrain', type=str, default='', help='pretrained_file')
parser.add_argument('--save', type=str, default='/tmp/xiaoya', help='path to save the final model')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=80, help='upper epoch limit')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--gpu_id', type=str, default='1', help='the gpu id to train language model')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--early_stopping_threshold', type=int, default=10, help='number of epoches to stop if validation ppl do not decrease')
parser.add_argument('--reload', action='store_true', help='reload data from files or load from cache(default)')
parser.add_argument('--lr', type=float, default=0.0008, help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=60, metavar='N', help='batch size')
parser.add_argument('--glyph_decay', type=float, default=0.8, help='decay of glyph_loss')
parser.add_argument('--glyph_ratio', type=float, default=0.01, help='ratio of glyph_loss, and 1-ratio is normal loss')
parser.add_argument('--rnn_type', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--nhid', type=int, default=1024, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.8, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cnn_dropout', type=float, default=0.5, help='dropout applied to cnn_embed')
parser.add_argument('--word_embsize', type=int, default=512, help='size of word embeddings')
parser.add_argument('--char_embsize', type=int, default=512, help='size of char embeddings')
parser.add_argument('--glyph_embsize', type=int, default=512, help='size of glyph embeddings')
parser.add_argument('--pretrained_word_embedding', type=str, default='', help='path to pretrained word embedding')
parser.add_argument('--pretrained_char_embedding', type=str, default='', help='path to pretrained char embedding')
parser.add_argument('--font_channels', type=int, default=1, metavar='N', help='number of channels for font')
parser.add_argument('--random_fonts', type=int, default=0, metavar='N', help='number of random fonts selected from font channels')
parser.add_argument('--font_size', type=int, default=12, help='fontsize for glyph')
parser.add_argument('--font_name', type=str, default='CJK/NotoSansCJKsc-Regular.otf', help='font name for glyph')
parser.add_argument('--use_traditional', action='store_true', help='use glyph of traditional word')
parser.add_argument('--font_normalize', action='store_true', help='use normalization for each font mask')
parser.add_argument('--subchar_type', type=str, default='', help='use pinyin or wubi encoding for subword')
parser.add_argument('--subchar_embsize', type=int, default=512, help='embedding size of each subword character')
parser.add_argument('--random_erase', action='store_true', help='random erase a rectangle from each image')
parser.add_argument('--num_fonts_concat', type=int, default=1, help='num of fonts concat together')
parser.add_argument('--glyph_cnn_type', type=str, default='Yuxian8', help='type of cnn')
parser.add_argument('--use_layer_norm', action='store_true', help='use layer_normalization after concatenation of two embeddings')
parser.add_argument('--use_batch_norm', action='store_true', help='use batch_normalization after concatenation of two embeddings')
parser.add_argument('--use_highway', action='store_true', help='use highway network after concatenation of two embeddings')
parser.add_argument('--yuxian_merge', action='store_true', help='use yuxian_merge')
parser.add_argument('--fc_merge', action='store_true', help='use fc_merge')
parser.add_argument('--use_maxpool', action='store_true', help='use maxpool')
parser.add_argument('--output_size', type=int, default=512, help='input size of lstm language model')
parser.add_argument('--level', type=str, default='word', help='word or char level of lm')
parser.add_argument('--char_drop', type=float, default=0.5, help='dropout applied to char_embedding')
parser.add_argument('--char2word_dim', type=int,  default=1024, help='dimension to char_embedding')
parser.add_argument('--glyph_groups', type=int,  default=16, help='groups of glyph cnn')


args = parser.parse_args()
args.loss_mask_ids = [1, 2]


save_path = os.path.join(args.save, datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
if not os.path.exists(save_path):
    os.makedirs(save_path)

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

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
torch.manual_seed(args.seed)


if args.reload:
    logger.info('Producing dataset...')
    corpus = WordCorpus(args.data)
    torch.save(corpus, os.path.join(args.data, 'word_corpus.pkl'))
    with open(os.path.join(args.data, 'word_dictionary.json'), 'w') as fo:
        json.dump({'idx2word': corpus.dictionary.idx2word, 'word2idx': corpus.dictionary.word2idx}, fo)
else:
    logger.info('Loading cached dataset...')
    corpus = torch.load(os.path.join(args.data, 'word_corpus.pkl'))
logger.info(F'train:{len(corpus.train[0])}\nvalid:{len(corpus.valid[0])}\ntest:{len(corpus.test[0])}')


def batchify(data_words, data_chars, bsz):
    """
    取训练集中所有的token，整理成batch_size个长Tensor
    :param data: 整个数据集的所有token
    :param bsz: batch_size
    :return: (long_seq_len, batch_size)
    """
    nbatch = data_words.size(0) // bsz
    data_words = data_words.narrow(0, 0, nbatch * bsz)
    data_chars = data_chars.narrow(0, 0, nbatch * bsz)
    data_words = data_words.view(bsz, -1).t().contiguous()
    data_chars = data_chars.view(bsz, -1).t().contiguous()
    return data_words.cuda(), data_chars.cuda()    #  n, b;  n*4, b


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):  # 普通RNN
        return h.detach()
    else:  # LSTM
        return [s.detach() for s in h]


def get_batch(source, idx):   #  n, b;  n*4, b
    """
    取source中第i个滑动窗口的数据作为data, 第i+1个滑动窗口的数据作为target, (seq_len, batch_size)
    :param source: 整理成batch的数据
    :param idx: 在长tensor上的index
    :return: 在长tensor上以index开头，最长为bptt，总共batch_size个样例组成的输入，以及下一个time_step作为gold
    """
    seq_len = min(args.bptt, len(source[0]) - 1 - idx)  # 长度一般是bptt，在结尾处委屈求全一下
    words = source[0][idx: idx + seq_len]
    chars = source[1][idx*4: (idx + seq_len)*4]
    target_words = source[0][idx + 1: idx + 1 + seq_len]
    return words, target_words, chars   #b,s; b,s; b, s*4


def train():
    best_val_loss = 1000
    nwords, nchars = len(corpus.dictionary.idx2word), len(corpus.dictionary.idx2char)
    args.idx2word = corpus.dictionary.idx2word
    args.idx2char = corpus.dictionary.idx2char
    train_data = batchify(data_words=corpus.train[0], data_chars=corpus.train[1], bsz=args.batch_size)  #  num_batches, batch_size;  n*4, b
    val_data = batchify(data_words=corpus.valid[0], data_chars=corpus.valid[1], bsz=args.batch_size)   #  n, b;  n*4, b
    if args.pretrain:
        model = torch.load(args.pretrain)
        # for m in model.glyph_embedder.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         for param in m.parameters():   # 停止训练cnn
        #             param.requires_grad = False
    else:
        model = GlyphEmbeddingForLM(model_config=args).cuda()
    logger.info(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=0, cooldown=3, factor=0.25, verbose=True, min_lr=1e-6,
                                  threshold=1e-4, eps=1e-6)
    logger.info('start training...')
    hidden = model.init_hidden(args.batch_size)
    epoch_start_time = time.time()
    best_epoch = 0

    for epoch in range(args.epochs):
        model.eval()  # 在validation上测试
        total_loss = 0.
        total_cnn_loss = 0.
        with torch.no_grad():
            for idx in range(0, val_data[0].size(0) - 1, args.bptt):
                data, targets, data_chars = get_batch(val_data, idx)
                output, hidden, glyph_classification_loss = model((data, data_chars), hidden)
                loss_lm = criterion(output.view(-1, nwords), targets.view(-1)).item()
                total_cnn_loss += len(data) * glyph_classification_loss
                total_loss += len(data) * loss_lm
                hidden = repackage_hidden(hidden)
                # print(data.shape, loss_lm, total_cnn_loss)
        val_loss = total_loss / len(val_data[0])
        total_cnn_loss /= len(val_data[0])
        if not isinstance(total_cnn_loss, float):
            total_cnn_loss = total_cnn_loss.item()
        best_val_loss = min(best_val_loss, val_loss)
        logger.info('-' * 100)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss lm:{:5.2f} cnn:{:5.4f} | valid ppl {:8.2f} | best valid ppl {:8.2f}'
              .format(epoch, (time.time() - epoch_start_time), val_loss, total_cnn_loss, math.exp(val_loss), math.exp(best_val_loss))) #
        logger.info('-' * 100)
        epoch_start_time = time.time()
        scheduler.step(val_loss)
        if val_loss == best_val_loss:  # Save the model if the validation loss is best so far.
            torch.save(model, os.path.join(save_path, 'model.pkl'))
            best_epoch = epoch

        model.train()  # 在training set上训练
        total_loss = 0.
        start_time = time.time()
        for i, idx in enumerate(range(0, train_data[0].size(0) - 1, args.bptt)):
            data, targets, data_chars = get_batch(train_data, idx)   #s,b  ; s, b   s*4, b
            hidden = repackage_hidden(hidden)
            model.zero_grad()  # 求loss和梯度
            output, hidden, glyph_classification_loss = model((data, data_chars), hidden)
            loss_lm = criterion(output.view(-1, nwords), targets.view(-1))
            loss = loss_lm * (1-args.glyph_ratio) + args.glyph_ratio * args.glyph_decay**(epoch+1) * glyph_classification_loss
            loss.backward()
            total_loss += loss_lm.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # 用梯度更新参数
            optimizer.step()

            if i % args.log_interval == 0 and i > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.4e} | ms/batch {:5.2f} |loss {:5.2f} | ppl {:8.2f}'
                      .format(epoch + 1, i, len(train_data[0]) // args.bptt, args.lr, elapsed * 1000 / args.log_interval,
                              cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        if epoch > best_epoch + args.early_stopping_threshold:
            break

    logger.info(math.exp(best_val_loss))
    with open(os.path.join(args.save, 'font.log'), 'a') as fo:
        fo.write(F'{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}\t{args}\t{math.exp(best_val_loss)}\n')


if __name__ == '__main__':
    train()
