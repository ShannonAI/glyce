#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch

import json 
import random 
import numpy as np
from PIL import ImageFont
from zhconv import convert


default_font_size = 12
default_font_path = os.path.join(root_path, "glyce/fonts")
default_font_name = 'cjk/NotoSansCJKsc-Regular.otf'
default_font = ImageFont.truetype(os.path.join(default_font_path, default_font_name), default_font_size)
font_list = ['bronzeware_script/HanYiShouJinShuFan-1.ttf', 'cjk/NotoSansCJKsc-Regular.otf', 'seal_script/方正小篆体.ttf', 
             'tablet_script/WenDingHuangYangJianWeiTi-2.ttf', 'regular_script/STKAITI.TTF', 'cursive_script/行草字体.ttf', 'clerical_script/STLITI.TTF',
             'cjk/STFANGSO.TTF', 'clerical_script/方正古隶繁体.ttf', 'regular_script/STXINGKA.TTF']



def get_font_names(mode='fixed', shuffle=False):
    if mode == 'fixed':
        return font_list
    simple_fonts = []
    all_fonts = []
    for folder in os.listdir(default_font_path):
        for i, font in enumerate(os.listdir(os.path.join(default_font_path, folder))):
            if i == 0:
                simple_fonts.append(folder + '/' + font)
            all_fonts.append(folder + '/' + font)
    if shuffle:
        random.shuffle(all_fonts)
    return simple_fonts if mode == 'simple' else all_fonts


def multiple_glyph_embeddings(num_fonts, chosen_font, idx2word, font_size=12, use_traditional=False, normalize=False):
    embeddings = []
    if num_fonts == 1:
        embeddings.append(vocab_glyph_embedding(idx2word, chosen_font, font_size, use_traditional, normalize))
    else:
        font_list = get_font_names()
        for i in range(num_fonts):
            print('handling', font_list[i])
            embeddings.append(vocab_glyph_embedding(idx2word, font_list[i], font_size, use_traditional, normalize))
    return torch.from_numpy(np.stack(embeddings, axis=1)).float()# 4400, 3, 13, 13


def vocab_glyph_embedding(idx2word, font_name='CJK/NotoSansCJKsc-Regular.otf', font_size=12, use_traditional=False, normalize=False):
    font = ImageFont.truetype(os.path.join(default_font_path, font_name), font_size)
    r = np.array([render_text_with_token_id(i, font, use_traditional, idx2word) for i in range(len(idx2word))])
    return (r - np.mean(r)) / np.std(r) if normalize else r


def render_text_with_token_id(token_id, font, use_traditional, idx2word):
    word = idx2word[token_id]
    word = convert(word, 'zh-hant') if use_traditional else word
    if len(word) > 1:
        return np.zeros((font.size + 1, font.size + 1))
    else:
        return pad_mask(render_text(word, font), font.size)


def render_text(text, font):
    mask = font.getmask(text)
    size = mask.size[::-1]
    a = np.asarray(mask).reshape(size)
    return a


def ascii_print(glyph_array):
    print('='*100)
    for l in glyph_array:
        char_line = ''
        for c in l:
            if c != 0:
                char_line += str(c % 2)
            else:
                char_line += ' '
        print(char_line)


def pad_mask(mask, fontsize):
    padded_mask = []
    for l in mask:
        padded_mask.append(l.tolist() + [0] * (fontsize + 1 - len(l)))
    for i in range(fontsize + 1 - len(padded_mask)):
        padded_mask.append([0]*(fontsize + 1))
    return np.array(padded_mask)[: fontsize + 1, : fontsize + 1]


if __name__ == '__main__':
    # feat = render_text('劃繁簡轉換後', default_font)
    # ascii_print(feat)
    # print(len(get_font_names()))
    with open(os.path.join('/data/ctb_v6/data/utf8/raw/', 'dictionary.json')) as fo:
        idx2word = json.load(fo)['idx2word']
    for folder in os.listdir(default_font_path):
        for font_name in os.listdir(os.path.join(default_font_path, folder)):
                font = ImageFont.truetype(os.path.join(default_font_path, folder, font_name), default_font_size)
                emb = vocab_glyph_embedding(os.path.join(default_font_path, folder, font_name), 24)
                assert emb.shape == (4401, 25, 25)
                useless = json.load(open('useless_font.json'))
                useless2 = json.load(open('useless_font2.json'))
                useless3 = json.load(open('useless_font3.json'))
                cnt = 0
                for idx in range(len(idx2word)):
                    feat = render_text_with_token_id(idx, font, False)
                    if feat.tolist() in [useless, useless2, useless3]:
                        print('useless')
                        cnt += 1
                    if idx % 1000 == 0:
                        ascii_print(feat)
                print('oov', cnt / len(idx2word), cnt, len(idx2word))
                if cnt / len(idx2word) > 0.2:
                    os.remove(os.path.join(default_font_path, folder, font_name))
