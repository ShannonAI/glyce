## GlyceConfig Descriptions 

Chinese Glyce Char/Word embeddings introduce new hyper-parameters in .config file 


parameter name | introduction | default value | type of variables 
----- | ------------- | --- | --- 
dropout | dropout rate of word/char embedding | 0.3 | float
cnn_dropout | dropout rate of group CNNs | 0.3 | float
idx2word | idx map to word tokens | - | dict 
idx2char | idx map to char  | - | dict 
word_embsize | size of Word-ID embedding | 1024 | int 
glyph_embsize | Glyph embedding size | - | int 
output_size | dimension size of final glyce embedding  | 256 | int 
char2word_dim | dimension size when converting char embeddings into word embeddings | 1024 | int 
pretrained_word_embedding | pretrained word embedding weight table | "" | list 
char_embsize | size of Char-ID embedding | 1024 | int 
pretrained_char_embedding | pretrained Chinese char embedding table | - | list 
pretrained_word_embedding | pretrained Chinese word embedding table | - | list 
font_size | script image size | 12 | int 
font_normalize | whether to normalize the gray value of images of the font | False | bool 
random_fonts | each batch randomly selects N different fonts | - | int 
font_channels | the number of fonts insert into the CNN channels. If random_fonts > 0, it represents the total number of fonts available for random selection in the font candidate pool | - | int 
font_name | the font name of the form "CJK/NotoSansCJKscReguar.otf" is valid only when font_channels=1 | - | str 
font_size | size of script image | 18 | int 
num_fonts_concat | concatenate the feature vectors obtained by passing the N fonts corresponding to one word through | 4 | int 
use_tranditional | whether to use traditional characters instead of simplified characters | False | bool 
subchar_type | whether to use pinyin("pinyin") or wubi ("wubi") | False | bool 
random_erase | whether randomly block a small piece of the gray scale image of Chinese scripts | False | bool  
glyph_cnn_type | the name of CNN model for extracting glyph information | yuxuanba | str 
use_batch_norm | whether or not to process the feature vector of images via batch norm | True | bool 
use_layer_norm | whether to use layer norm | False | bool 
use_highway | whether to pass concated vectors to highway connection | False | bool 
use_max_pool | use max-pooling or fc merge to convert glyce char vectors into glyph word vectors | False | bool
yuxian_merge | whether to pass concated vectors into yuxian_merge | False | bool 
fc_merge | whether or not to pass concated vectors to MLP | True | bool 
level | input granularity e.g. Chinese char or word | char | str
char_drop | dropout rate for char ID embedding | 0.3 | float 
glyph_groups | number of groups for glyph CNNs | 16 | int 
loss_mask_ids | word idxs in order to mask when calculates glyph loss ("[CLS]","UNK" ID) | (0, 1) | list[int]

