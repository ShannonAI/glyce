import collections
import pickle
import os

def make_char_vocab(filename, output_path):

    chars = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()

    words = data.strip().split('\n')
    print(len(words))

    for word in words[4:]: # ignore ['<PAD>', '<UNK>', '<ROOT>', '<NUM>']
        for char in word:
            chars.append(char)

    chars_counter = collections.Counter(chars).most_common()

    char_vocab = {'<PAD>', '<UNK>', '<ROOT>', '<NUM>'}
    char_vocab = ['<PAD>', '<UNK>', '<ROOT>', '<NUM>'] + [item[0] for item in chars_counter]

    print(char_vocab)

    char_to_idx = {char:idx for idx, char in enumerate(char_vocab)}
    idx_to_char = {idx:char for idx, char in enumerate(char_vocab)}

    print(char_to_idx)
    
    vocab_path = os.path.join(output_path,'char.vocab')
    char2idx_path = os.path.join(output_path,'char2idx.bin')
    idx2char_path = os.path.join(output_path,'idx2char.bin')

    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(char_vocab))

    with open(char2idx_path, 'wb') as f:
        pickle.dump(char_to_idx, f)

    with open(idx2char_path, 'wb') as f:
        pickle.dump(idx_to_char, f)


make_char_vocab('temp/word.vocab', 'temp')