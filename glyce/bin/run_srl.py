#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Muyu LI
# Last update: 2019.03.12 
# First create: 2019.03.12 
# Description:
# run_srl.py 


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path) 


import torch 
from torch import nn
from torch import optim


import time
import argparse 
import _pickle as pickle 
from tqdm import tqdm 


from glyce import GlyceConfig 
from glyce import WordGlyceEmbedding 
import glyce.models.srl.model as model
import glyce.models.srl.data_utils as data_utils 
import glyce.models.srl.inter_utils as inter_utils 
from glyce.models.srl.utils import USE_CUDA
from glyce.models.srl.utils import get_torch_variable_from_np, get_data
from glyce.models.srl.scorer import eval_train_batch, pruning_eval_data
from glyce.models.srl.data_utils import *


from allennlp.modules.elmo import Elmo
from allennlp.data import dataset as Dataset 
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer






def get_elmo(options_file, weight_file):

    # Create the ELMo class.  This example computes two output representation
    # layers each with separate layer weights.
    # We recommend adding dropout (50% is good default) either here or elsewhere
    # where ELMo is used (e.g. in the next layer bi-LSTM).
    elmo = Elmo(options_file, weight_file, num_output_representations=2,
                do_layer_norm=False, dropout=0.2)

    if USE_CUDA:
        elmo.cuda()

    return elmo

def seed_everything(seed, cuda=False):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)


def make_parser():

    parser = argparse.ArgumentParser(description='PyTorch SRL with K-pruning algorithm')

    # input
    parser.add_argument('--train_data', type=str, help='Train Dataset with CoNLL09 format')
    parser.add_argument('--valid_data', type=str, help='Train Dataset with CoNLL09 format')
    parser.add_argument('--test_data', type=str, help='Train Dataset with CoNLL09 format')
    parser.add_argument('--ood_data', type=str, help='OOD Dataset with CoNLL09 format')

    parser.add_argument('--K', type=int, default=0, help='the K in K-pruning algorithm')
    parser.add_argument('--seed', type=int, default=100, help='the random seed')
    parser.add_argument('--unify_pred', action='store_true',
                        help='[USE] unify the predicate classification')

    # this default value is from PATH LSTM, you can just follow it too
    # if you want to do the predicate disambiguation task, you can replace the accuracy with yours.
    parser.add_argument('--dev_pred_acc', type=float, default=0.9477,
                            help='Dev predicate disambiguation accuracy')
    parser.add_argument('--test_pred_acc', type=float, default=0.9547,
                            help='Test predicate disambiguation accuracy')
    parser.add_argument('--ood_pred_acc', type=float, default=0.8618,
                            help='OOD predicate disambiguation accuracy')

    # preprocess
    parser.add_argument('--preprocess', action='store_true',
                         help='Preprocess')
    parser.add_argument('--tmp_path', type=str, help='temporal path')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--result_path', type=str, help='result path')
    parser.add_argument('--pretrain_embedding', type=str, help='Pretrain embedding like GloVe or word2vec')
    parser.add_argument('--pretrain_emb_size', type=int, default=100,
                        help='size of pretrain word embeddings')

    # train 
    parser.add_argument('--train', action='store_true',
                            help='Train')
    parser.add_argument('--epochs', type=int, default=20,
                            help='Train epochs')
    parser.add_argument('--dropout', type=float, default=0.1,
                            help='Dropout when training')
    parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                            help='Batch size in train and eval')
    parser.add_argument('--word_emb_size', type=int, default=100,
                            help='Word embedding size')
    parser.add_argument('--pos_emb_size', type=int, default=32,
                            help='POS tag embedding size')
    parser.add_argument('--lemma_emb_size', type=int, default=100,
                            help='Lemma embedding size')
    parser.add_argument('--use_deprel', action='store_true',
                        help='[USE] dependency relation')
    parser.add_argument('--deprel_emb_size', type=int, default=64,
                            help='Dependency relation embedding size')
    parser.add_argument('--bilstm_hidden_size', type=int, default=512,
                            help='Bi-LSTM hidden state size')
    parser.add_argument('--bilstm_num_layers', type=int, default=4,
                            help='Bi-LSTM layer number')
    parser.add_argument('--valid_step', type=int, default=1000,
                            help='Valid step size')
    parser.add_argument('--use_highway', action='store_true',
                        help='[USE] highway connection')
    parser.add_argument('--highway_num_layers',type=int, default=10,
                            help='Highway layer number')
    parser.add_argument('--use_flag_emb', action='store_true',
                        help='[USE] predicate flag embedding')
    parser.add_argument('--flag_emb_size',type=int, default=16,
                            help='Predicate flag embedding size')
    parser.add_argument('--use_elmo', action='store_true',
                        help='[USE] ELMo embedding')
    parser.add_argument('--elmo_emb_size',type=int, default=300,
                            help='ELMo embedding size')
    parser.add_argument('--elmo_options',type=str,
                            help='ELMo options file')
    parser.add_argument('--elmo_weight',type=str,
                            help='ELMo weight file')
    parser.add_argument('--clip', type=float, default=0,
                        help='gradient clipping')

    # eval
    parser.add_argument('--eval', action='store_true',
                            help='Eval')
    parser.add_argument('--model', type=str, help='Model')


    # glyph
    parser.add_argument('--glyph_ratio', type=float, default=0.01, help='glyph classification loss ratio')
    parser.add_argument('--glyph_dropout', type=float, default=0.5, help='glyph dropout')
    parser.add_argument('--glyph_cnn_dropout', type=float, default=0.5, help='glyph cnn dropout rate')
    parser.add_argument('--glyph_char_dropout', type=float, default=0.5, help='glyph char dropout rate')
    parser.add_argument('--glyph_font_channels', type=int, default=1, help='glyph font channels.')
    parser.add_argument('--glyph_embsize', type=int, default=256, help='glyph embedding size')
    parser.add_argument('--glyph_char2word_dim', type=int, default=256, help='glyph char2word size')
    parser.add_argument('--glyph_output_size', type=int, default=256, help='glyph output size')
    parser.add_argument('--glyph_char_embsize', type=int, default=256, help='glyph char embedding size')

    return parser


if __name__ == '__main__':

    print('Semantic with K-pruning algorithm')

    args = make_parser().parse_args()

    print(args)

    # set random seed
    # seed_everything(args.seed, USE_CUDA)

    unify_pred = args.unify_pred

    # do preprocessing
    # if args.preprocess:
    train_file = args.train_data
    dev_file = args.valid_data
    test_file = args.test_data
    test_ood_file = args.ood_data

    tmp_path = args.tmp_path

    if tmp_path is None:
        print('Fatal error: tmp_path cannot be None!')
        exit()

    print('start preprocessing data...')

    

    if args.preprocess:
        start_t = time.time()

        # make word/pos/lemma/deprel/argument vocab
        print('\n-- making (word/lemma/pos/argument/predicate) vocab --')
        vocab_path = tmp_path
        print('word:')
        make_word_vocab(train_file,vocab_path, unify_pred=unify_pred)
        print('pos:')
        make_pos_vocab(train_file,vocab_path, unify_pred=unify_pred)
        print('lemma:')
        make_lemma_vocab(train_file,vocab_path, unify_pred=unify_pred)
        print('deprel:')
        make_deprel_vocab(train_file,vocab_path, unify_pred=unify_pred)
        print('argument:')
        make_argument_vocab(train_file, dev_file, test_file, vocab_path, unify_pred=unify_pred)
        print('predicate:')
        make_pred_vocab(train_file, dev_file, test_file, vocab_path)

        deprel_vocab = load_deprel_vocab(os.path.join(tmp_path, 'deprel.vocab'))

        # shrink pretrained embeding
        print('\n-- shrink pretrained embeding --')
        pretrain_file = args.pretrain_embedding
        pretrained_emb_size = args.pretrain_emb_size
        pretrain_path = tmp_path
        shrink_pretrained_embedding(train_file, dev_file, test_file, pretrain_file, pretrained_emb_size, pretrain_path)

        # make dataset input
        if args.K == 0:
            make_dataset_input(train_file, os.path.join(tmp_path,'train.input'), unify_pred=unify_pred, deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(tmp_path,'train.pickle.input'))
            make_dataset_input(dev_file, os.path.join(tmp_path,'dev.input'), unify_pred=unify_pred, deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(tmp_path,'dev.pickle.input'))
            make_dataset_input(test_file, os.path.join(tmp_path,'test.input'), unify_pred=unify_pred, deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(tmp_path,'test.pickle.input'))
            if test_ood_file is not None:
                make_dataset_input(test_ood_file, os.path.join(tmp_path,'test_ood.input'), unify_pred=unify_pred, deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(tmp_path,'test_ood.pickle.input'))
        else:
            make_k_order_pruning_dataset_input(train_file, os.path.join(tmp_path,'train.input'), args.K, unify_pred=unify_pred, deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(tmp_path,'train.pickle.input'))
            make_k_order_pruning_dataset_input(dev_file, os.path.join(tmp_path,'dev.input'), args.K, unify_pred=unify_pred, deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(tmp_path,'dev.pickle.input'))
            make_k_order_pruning_dataset_input(test_file, os.path.join(tmp_path,'test.input'), args.K, unify_pred=unify_pred, deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(tmp_path,'test.pickle.input'))
            if test_ood_file is not None:
                make_k_order_pruning_dataset_input(test_ood_file, os.path.join(tmp_path,'test_ood.input'), args.K, unify_pred=unify_pred, deprel_vocab=deprel_vocab, pickle_dump_path=os.path.join(tmp_path,'test_ood.pickle.input'))

        print('\t data preprocessing finished! consuming {} s'.format(int(time.time()-start_t)))
    
    print('\t start loading data...')

    start_t = time.time()

    train_input_file = os.path.join(os.path.dirname(__file__),'temp/train.pickle.input')
    dev_input_file = os.path.join(os.path.dirname(__file__),'temp/dev.pickle.input')
    test_input_file = os.path.join(os.path.dirname(__file__),'temp/test.pickle.input')
    if test_ood_file is not None:
        test_ood_input_file = os.path.join(os.path.dirname(__file__),'temp/test_ood.pickle.input')

    # train_dataset = data_utils.load_dataset_input(train_input_file)
    # dev_dataset = data_utils.load_dataset_input(dev_input_file)
    # test_dataset = data_utils.load_dataset_input(test_input_file)
    # if test_ood_file is not None:
    #     test_ood_dataset = data_utils.load_dataset_input(test_ood_input_file)

    train_data = data_utils.load_dump_data(train_input_file)
    assert train_data['K'] == args.K
    dev_data = data_utils.load_dump_data(dev_input_file)
    assert dev_data['K'] == args.K
    test_data = data_utils.load_dump_data(test_input_file)
    assert test_data['K'] == args.K
    if test_ood_file is not None:
        test_ood_data = data_utils.load_dump_data(test_ood_input_file)
        assert test_ood_data['K'] == args.K

    train_dataset = train_data['input_data']
    dev_dataset = dev_data['input_data']
    test_dataset = test_data['input_data']
    if test_ood_file is not None:
        test_ood_dataset = test_ood_data['input_data']

    word2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/word2idx.bin'))
    idx2word = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2word.bin'))

    char2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/char2idx.bin'))
    idx2char = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2char.bin'))
    
    lemma2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/lemma2idx.bin'))
    idx2lemma = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2lemma.bin'))

    pos2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/pos2idx.bin'))
    idx2pos = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2pos.bin'))

    deprel2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/deprel2idx.bin'))
    idx2deprel = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2deprel.bin'))

    pretrain2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/pretrain2idx.bin'))
    idx2pretrain = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2pretrain.bin'))

    argument2idx = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/argument2idx.bin'))
    idx2argument = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/idx2argument.bin'))

    pretrain_emb_weight = data_utils.load_dump_data(os.path.join(os.path.dirname(__file__),'temp/pretrain.emb.bin'))

    print('\t data loading finished! consuming {} s'.format(int(time.time()-start_t)))

    result_path = args.result_path

    print('\t start building model...')
    start_t = time.time()

    dev_predicate_sum = dev_data['predicate_sum']
    test_predicate_sum = test_data['predicate_sum']
    if test_ood_file is not None:
        test_ood_predicate_sum = test_ood_data['predicate_sum']

    dev_out_of_pruning = dev_data['out_of_pruning_sum']
    test_out_of_pruning = test_data['out_of_pruning_sum']
    if test_ood_file is not None:
        test_ood_out_of_pruning = test_ood_data['out_of_pruning_sum']
    
    dev_predicate_correct = int(dev_predicate_sum * args.dev_pred_acc)
    test_predicate_correct = int(test_predicate_sum * args.test_pred_acc)
    if test_ood_file is not None:
        test_ood_predicate_correct = int(test_ood_predicate_sum * args.ood_pred_acc)

    print('dev predicate sum:{} target sum:{} target pruning:{} argument sum:{} argument pruning:{}'.format(dev_data['predicate_sum'], dev_data['target_sum'], dev_data['out_of_target_sum'], dev_data['argument_sum'], dev_data['out_of_pruning_sum']))
    print('test predicate sum:{} target sum:{} target pruning:{} argument sum:{} argument pruning:{}'.format(test_data['predicate_sum'], test_data['target_sum'], test_data['out_of_target_sum'], test_data['argument_sum'], test_data['out_of_pruning_sum']))
    if test_ood_file is not None:
        print('test predicate sum:{} target sum:{} target pruning:{} argument sum:{} argument pruning:{}'.format(test_ood_data['predicate_sum'], test_ood_data['target_sum'], test_ood_data['out_of_target_sum'], test_ood_data['argument_sum'], test_ood_data['out_of_pruning_sum']))

    # hyper parameters
    max_epoch = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    dropout = args.dropout
    word_embedding_size = args.word_emb_size
    pos_embedding_size = args.pos_emb_size
    pretrained_embedding_size = args.pretrain_emb_size
    lemma_embedding_size = args.lemma_emb_size
    

    bilstm_hidden_size = args.bilstm_hidden_size
    bilstm_num_layers = args.bilstm_num_layers
    show_steps = args.valid_step

    use_highway = args.use_highway
    highway_layers = args.highway_num_layers

    use_flag_embedding = args.use_flag_emb
    flag_embedding_size = args.flag_emb_size

    use_elmo = args.use_elmo
    elmo_embedding_size = args.elmo_emb_size
    elmo_options_file = args.elmo_options
    elmo_weight_file = args.elmo_weight
    elmo = None
    if use_elmo:
        elmo = get_elmo(elmo_options_file, elmo_weight_file)
    
    use_deprel = args.use_deprel
    deprel_embedding_size = args.deprel_emb_size
    
    if args.train:
        FLAG = 'TRAIN'
    if args.eval:
        FLAG = 'EVAL'
        MODEL_PATH = args.model

    if FLAG == 'TRAIN':
        model_params = {
            "dropout":dropout,
            "batch_size":batch_size,
            "word_vocab_size":len(word2idx),
            "lemma_vocab_size":len(lemma2idx),
            "pos_vocab_size":len(pos2idx),
            "deprel_vocab_size":len(deprel2idx),
            "pretrain_vocab_size":len(pretrain2idx),
            "word_emb_size":word_embedding_size,
            "lemma_emb_size":lemma_embedding_size,
            "pos_emb_size":pos_embedding_size,
            "pretrain_emb_size":pretrained_embedding_size,
            "pretrain_emb_weight":pretrain_emb_weight,
            "bilstm_num_layers":bilstm_num_layers,
            "bilstm_hidden_size":bilstm_hidden_size,
            "target_vocab_size":len(argument2idx),
            "use_highway":use_highway,
            "highway_layers": highway_layers,
            "use_deprel":use_deprel,
            "deprel_emb_size":deprel_embedding_size,
            "flag_embedding_size":flag_embedding_size,
            'use_elmo':use_elmo,
            "elmo_embedding_size":elmo_embedding_size,
            "use_flag_embedding":use_flag_embedding,
            "elmo_options_file":elmo_options_file,
            "elmo_weight_file":elmo_weight_file,
            "deprel2idx":deprel2idx
        }

        ratio = args.glyph_ratio


        # build model
        dic = {
        'dropout': args.glyph_dropout,     # dropout of word_embedding
        'cnn_dropout':args.glyph_cnn_dropout , # glyph cnn dropout rate
        'char_drop': args.glyph_char_dropout,   # dropout applied to char_embedding
        'font_channels': args.glyph_font_channels, 
        'glyph_embsize': args.glyph_embsize,
        'char2word_dim': args.glyph_char2word_dim, 
        'output_size': args.glyph_output_size, 
        'char_embsize': args.glyph_char_embsize, 
        # the above parameters are important 

        'idx2word': idx2word,
        'idx2char': idx2char,
        'font_size': 12,
        'word_embsize': 128,
        'random_fonts': 0, 
        'font_name': 'CJK/NotoSansCJKsc-Regular.otf', 
        'use_traditional': False,
        'font_normalize': False, 
        'subchar_type': '', 
        'random_erase': False, 
        'num_fonts_concat': 1, 
        'glyph_cnn_type': 'Yuxian8',
        'use_batch_norm': True, 
        'use_layer_norm': False,
        'use_highway': False, 
        'yuxian_merge': False, 
        'fc_merge': False, 
        'level': 'word',
        'use_maxpool': True, 
        'glyph_groups': 16}

        config = GlyceConfig(dic)
        # config = DefualtGlyceConfig.from_dict(dic)
        # config = GlyphEmbeddingConfig.from_dict(dic)
        srl_model = model.End2EndModel(model_params, config)

        if USE_CUDA:
            srl_model.cuda()

        ignored_params = list(map(id, srl_model.glyph.glyph_embeddings.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, srl_model.parameters())


        criterion = nn.CrossEntropyLoss()

        
        optimizer = optim.Adam(base_params, lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.97)

        print(srl_model)

        print('\t model build finished! consuming {} s'.format(int(time.time()-start_t)))

        print('\nStart training...')

        dev_best_score = None
        test_best_score = None
        test_ood_best_score = None
        
        for epoch in range(max_epoch):

            epoch_start = time.time()
            for batch_i, train_input_data in enumerate(inter_utils.get_batch(train_dataset, batch_size, word2idx, char2idx,
                                                                    lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, shuffle=True, drop_last=True)):
                
                target_argument = train_input_data['argument']
                
                flat_argument = train_input_data['flat_argument']

                target_batch_variable = get_torch_variable_from_np(flat_argument)

                bs = train_input_data['batch_size']
                sl = train_input_data['seq_len']
                
                out, glyph_loss = srl_model(train_input_data, elmo)

                loss = criterion(out, target_batch_variable)
                loss = (1 - ratio) * loss + ratio * glyph_loss

                optimizer.zero_grad()
                loss.backward()
                if args.clip > 0:
                    nn.utils.clip_grad_norm(srl_model.parameters(), args.clip)
                #torch.nn.init.orthogonal
                optimizer.step()
                
                if batch_i > 0 and batch_i % show_steps == 0: 

                    _, pred = torch.max(out, 1)

                    pred = get_data(pred)

                    #pred = pred.reshape([bs, sl])

                    print('\n')
                    print('*'*80)

                    eval_train_batch(epoch, batch_i, loss.data[0], flat_argument, pred, argument2idx)

                    print('dev:')
                    score, dev_output = pruning_eval_data(srl_model, elmo, dev_dataset, batch_size, dev_out_of_pruning, word2idx, char2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2argument, unify_pred, dev_predicate_correct, dev_predicate_sum)
                    if dev_best_score is None or score[2] > dev_best_score[2]:
                        dev_best_score = score
                        torch.save(srl_model, os.path.join(args.model_path,'{:.2f}.pkl'.format(dev_best_score[2]*100)))
                    print('\tdev best P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(dev_best_score[0] * 100, dev_best_score[1] * 100,
                                                                                                    dev_best_score[2] * 100, dev_best_score[3] * 100,
                                                                                                    dev_best_score[4] * 100, dev_best_score[5] * 100))

                    scheduler.step()

                ratio = ratio * 0.8

                print('\repoch {} batch {} consume:{} s'.format(epoch, batch_i, int(time.time()-epoch_start)), end="")
                epoch_start = time.time()

                

    else:
        srl_model = torch.load(MODEL_PATH)
        srl_model.eval()
        print('test:')
        score, test_output = pruning_eval_data(srl_model, elmo, test_dataset, batch_size, test_out_of_pruning, word2idx, char2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2argument, unify_pred, test_predicate_correct, test_predicate_sum)
        output_predict(os.path.join(result_path,'test_argument_{:.2f}.pred'.format(score[2]*100)),test_output)
        # print('\ttest P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(score[0] * 100, score[1] * 100,
        #                                                                                 score[2] * 100, score[3] * 100,
        #                                                                                 score[4] * 100, score[5] * 100))

        if test_ood_file is not None: 
            print('ood:')
            score, ood_output = pruning_eval_data(srl_model, elmo, test_ood_dataset, batch_size, test_ood_out_of_pruning, word2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2argument, unify_pred, test_ood_predicate_correct, test_ood_predicate_sum)
            output_predict(os.path.join(result_path,'ood_argument_{:.2f}.pred'.format(score[2]*100)),ood_output)
            # print('\tood P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(score[0] * 100, score[1] * 100,
            #                                                                                 score[2] * 100, score[3] * 100,
            #                                                                                 score[4] * 100, score[5] * 100))


