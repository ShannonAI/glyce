# Glyce: Glyph-vectors for Chinese Character Representations 


**Glyce** is an open-source toolkit built on top of PyTorch and is developed by [Shannon.AI](http://www.shannonai.com). 



## Citation 

To appear in NeurIPS 2019. 

[Glyce: Glyph-vectors for Chinese Character Representations](https://arxiv.org/abs/1901.10125)

(Yuxian Meng*, Wei Wu*, Fei Wang*, Xiaoya Li*, Ping Nie, Fan Yin, Muyu Li, Qinghong Han, Xiaofei Sun and Jiwei Li, 2019)

```

@article{meng2019glyce,
  title={Glyce: Glyph-vectors for Chinese Character Representations},
  author={Meng, Yuxian and Wu, Wei and Wang, Fei and Li, Xiaoya and Nie, Ping and Yin, Fan and Li, Muyu and Han, Qinghong and Sun, Xiaofei and Li, Jiwei},
  journal={arXiv preprint arXiv:1901.10125},
  year={2019}
}
```

<div width="20%", height="20%", align="center">
   <img src="https://github.com/stepperL/glyce_pngs/blob/master/glyce1.0_pngs/glyce1_overview.png"><br><br>
</div>



## Table of Contents 

- [What is Glyce ?](#What-is-Glyce-?)
- [Experiment Results](#Experiment-Results)
    - [Sequence Labeling Tasks](#1.-Sequence-Labeling-Tasks)
    - [Sentence Pair Classification](#2.-Sentence-Pair-Classification)
    - [Single Sentence Classification](#3.-Single-Sentence-Classification)
    - [Chinese Semantic Role Labeling](#4.-Chinese-Semantic-Role-Labeling)
    - [Chinese Dependency Parsing](#5.-Chinese-Dependency-Parsing)
- [Requirements](#Requirements)
- [Installation](#Installation) 
- [Quick Start of Glyce](#Quick-Start-of-Glyce)
    - [Usage of Glyce Char/Word Embedding](#Usage-of-Glyce-Char/Word-Embedding)
    - [Usage of Glyce-BERT](#Usage-of-Glyce-BERT)
- [Folder Description](#Folder-Description)
- [Implementation](#Implementation)
- [Download Task Data](#Download-Task-Data)
- [Welcome Contributions to Glyce Open Source Project](#Welcome-Contributions-to-Glyce-Open-Source-Project )
- [Acknowledgement](#Ackowledgement) 
- [Contact](#Contact)





## What is Glyce ?


Glyce is a Chinese char representation based on Chinese glyph information. Glyce Chinese char embeddings are composed by two parts: (1) glyph-embeddings and (2) char-ID embeddings. The two parts are combined using concatenation, a highway network or a fully connected layer. Glyce word embeddings are also composed by two parts: (1) word glyph-embeddings and (2) word-ID embeddings. Glyph word embedding is the output of forwarding glyph embeddings for each char in the word through the max-pooling layer. Then glyph word embedding combine word-ID embedding using concatenation to get glyce word embedding. 


Here are the some highlights in glyce: 

#### 1. Utilize Chinese Logographic Information 
Glyce utilizes useful logographic information by encoding images of historical and contemporary scripts, along with the scripts of different writing styles. 


#### 2. Combine Glyce with Chinese Pre-trained BERT Model
We combine Glyce with Pre-trained Chinese BERT model and adopt specific layer to downstream tasks. The Glyce-BERT model outperforms BERT and sets new SOTA results for tagging (NER, CWS, POS), sentence pair classification, single sentence classification tasks. 


#### 3. Propose Tianzige-CNN(田字格) to Model Chinese Char
We propose the Tianzige-CNN(田字格) structure, which is tailored to Chinese character modeling. Tianzige-CNN(田字格) tackle the issue of small number of Chinese characters and small sacle of image compared to Imagenet. 


#### 4. Auxiliary Task Performs As a Regularizer
During training process, image-classification loss performs as an auxiliary training objective with the purpose of preventing overfitting and promiting the model's ability to generalize. 




## Experiment Results 

### 1. Sequence Labeling Tasks
#### Named Entity Recognition (NER)


MSRA(Levow, 2006), OntoNotes 4.0(Weischedel et al., 2011), Resume(Zhang et al., 2018). 



Model | P(onto) | R(onto) | F1(onto) | P(msra) | R(msra) | F1(msra)  
---------- | ------ | ------ | ------ | ------ | ------ | ------  
CRF-LSTM | 74.36 | 69.43 | 71.81 | 92.97 | 90.62 | 90.95 
Lattice-LSTM | 76.35 | 71.56 | 73.88  | 93.57 | 92.79 | 93.18 
Lattice-LSTM+Glyce | 82.06 | 68.74 | 74.81 | 93.86 | 93.92 | 93.89  
| | | | **(+0.93)** | | |  **(+0.71)**
BERT | 78.01 | 80.35 | 79.16 | 94.97 | 94.62 | 94.80
ERNIE | - | - | - | - | - | 93.8 
Glyce+BERT | 81.87 | 81.40 | 80.62 | 95.57 | 95.51 | 95.54
| | | |  **(+1.46)** | | | **(+0.74)**



Model | P(resume) | R(resume) | F1(resume) | P(weibo) | R(weibo) | F1(weibo)
---------- | ------ | ------ | ------  |  ------ | ------ | ------ 
CRF-LSTM | 94.53 | 94.29 | 94.41 | 51.16 | 51.07 | 50.95 
Lattice-LSTM | 94.81 | 94.11 | 94.46  | 52.71 | 53.92 | 53.13 
Lattice-LSTM+Glyce | 95.72 | 95.63 | 95.67  | 53.69 | 55.30 | 54.32 
| | | | **(+1.21)** | | | **(+1.19)**
BERT | 96.12 | 95.45 | 95.78 | 67.12 | 66.88 | 67.33 
Glyce+BERT | 96.62 | 96.48 | 96.54 | 67.68 | 67.71 | 67.60 
| | | | **(+0.76)** | | | **(+0.76)**

 

#### Chinese Part-Of-Speech Tagging (POS)

CTB5, CTB6, CTB9 and UD1  



Model | P(ctb5) | R(ctb5) | F1(ctb5) |  P(ctb6) | R(ctb6) | F1(ctb6)
---------- | ------ | ------ | ------ | ------ | ------ | ------
(Shao, 2017) (sig) | 93.68 | 94.47 | 94.07 | - | - | 90.81 
(Shao, 2017) (ens) | 93.95 | 94.81 | 94.38  | - | - | 
Lattice-LSTM | 94.77 | 95.51 | 95.14 | 92.00 | 90.86 | 91.43 
Glyce+Lattice-LSTM | 95.49 | 95.72 | 95.61   | 92.72 | 91.14 | 91.92 
| | | | **(+0.47)** | | | **(+0.49)**
BERT | 95.86 | 96.26 | 96.06  | 94.91 | 94.63 | 94.77 
Glyce+BERT | 96.50 | 96.74 | 96.61 | 95.56 | 95.26 | 95.41
| | | | **(+0.55)** | | | **(+0.55)**



Model | P(ctb9) | R(ctb9) | F1(ctb9) | P(ud1) | R(ud1) | F1(ud1)  
---------- | ------ | ------ | ------ | ------ | ------ | ------ 
(Shao, 2017) (sig) | 91.81 | 94.47 | 91.89  | 89.28 | 89.54 | 89.41  
(Shao, 2017) (ens) | 92.28 | 92.40 | 92.34  | 89.67 | 89.86 | 89.75 
Lattice-LSTM | 92.53 | 91.73 | 92.13 | 90.47 | 89.70 | 90.09 
Glyce+Lattice-LSTM | 92.28 | 92.85 | 92.38   | 91.57 | 90.19 | 90.87
| | | | **(+0.25)** | | | **(+0.78)** 
BERT | 92.43 | 92.15 | 92.29 | 95.42 | 94.97 | 95.19 
Glyce+BERT | 93.49 | 92.84 | 93.15 | 96.19 | 96.10 | 96.14 
| | | | **(+0.86)** | | | **(+1.35)**




#### Chinese Word Segmentation (CWS)

PKU, CITYU, MSR and AS. 


Model | P(pku) | R(pku) | F1(pku) | P(cityu) | R(cityu) | F1(cityu)  
---------- | ------ | ------ | ------ | ------ | ------ | ------ 
BERT | 96.8 | 96.3 | 96.5 | 97.5 | 97.7 | 97.6
Glyce+BERT | 97.1 | 96.4 | 96.7 | 97.9 | 98.0 | 97.9
| | | | **(+0.2)** | | | **(+0.3)**


Model | P(msr) | R(msr) | F1(msr) | P(as) | R(as) | F1(as)  
---------- | ------ | ------ | ------ | ------ | ------ | ------ 
BERT | 98.1 | 98.2 | 98.1 | 96.7 | 96.4 | 96.5
Glyce+BERT | 98.2 | 98.3 | 98.3 | 96.6 | 96.8 | 96.7
| | | | **(+0.2)** | | | **(+0.2)**



### 2. Sentence Pair Classification

#### Dataset Description 

The BQ corpus, XNLI, LCQMC, NLPCC-DBQA



Model | P(bq) | R(bq) | F1(bq) | Acc(bq) | P(lcqmc) | R(lcqmc) | F1(lcqmc) | Acc(lcqmc)  
-------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------
BiMPM | 82.3 | 81.2 | 81.7 | 81.9 | 77.6 | 93.9 | 85.0 | 83.4
BiMPM+Glyce | 81.9 | 85.5 | 83.7 | 83.3 | 80.4 | 93.4 | 86.4 | 85.3 
| | | | **(+2.0)** | **(+1.4)** | | | **(+1.4)** | **(+1.9)**
BERT | 83.5 | 85.7 | 84.6 | 84.8 | 83.2 | 94.2 | 88.2 | 87.5 
ERNIE | - | - | - | - | - | - | - | 87.4 
Glyce+BERT | 84.2 | 86.9 | 85.5 | 85.8 | 86.8 | 91.2 | 88.8 | 88.7
| | | | **(+0.9)** | **(+1.0)** | | | **(+0.6)** | **(+1.2)** 


Model | Acc(xnli) | P(nlpcc-dbqa) | R(nlpcc-dbqa) | F1(nlpcc-dbqa)
-------- | -------- | ------ | ------ | ------
BIMPM | 67.5 | 78.8 | 56.5 | 65.8 
BIMPM+Glyce | 67.7  | 76.3 | 59.9 | 67.1 
| | **(+0.2)** | | | **(+1.3)**
BERT | 78.4  | 79.6 | 86.0 | 82.7 
ERNIE | 78.4 | - | - | 82.7
Glyce+BERT | 79.2 | 81.1 | 85.8 | 83.4 
| | **(+0.8)**  | | | **(+0.7)**




###  3.Single Sentence Classification

#### Dataset Description 

ChnSentiCorp, Fudan and Ifeng. 

Model | ChnSentiCorp | Fudan | iFeng  
-------------- | ------ | ------ | ------ 
LSTM | 91.7 | 95.8 | 84.9 
LSTM+Glyce | 93.1 | 96.3 | 85.8  
| | **(+1.4)** | **(+0.5)** | **(+0.9)**
BERT | 95.4 | 99.5 | 87.1 
ERNIE | 95.4 | - | - 
Glyce+BERT | 95.9 | 99.8 | 87.5 
| | **(+0.5)** | **(+0.3)** | **(+0.4)**


### 4. Chinese Semantic Role Labeling

#### Dataset Description 

CoNLL-2009 

#### Model Performance 

Model | Precision | Recall | F1  
-------------- | ------ | ------ | ------  
Roth and Lapata (2016) | 76.9 | 73.8 | 75.3 
Marcheggiani and Titov (2017) | 84.6 | 80.4 | 82.5 
K-order pruning (He et al., 2018) | 84.2 | 81.5 | 82.8  
K-order pruning + Glyce-word | 85.4 | 82.1 | 83.7
| | **(+0.8)** | **(+0.6)** | **(+0.9)**


### 5. Chinese Dependency Parsing 

#### Dataset Description 

Chinese Penn TreeBank 5.1.  Dataset splits follows (Dozat and Manning, 2016). 

#### Model Performance 


Model | UAS | LAS  
-------------- | ------ | ------
Ballesteros et al. (2016) | 87.7 | 86.2 
Kiperwasser and Goldberg (2016) | 87.6 | 86.1  
Cheng et al. (2016) | 88.1 | 85.7  
Biaffine | 89.3 | 88.2 
Biaffine+Glyce-word | 90.2 | 89.0 
| | **(+0.9)** | **(+0.8)**



## Requirements 

- Python Version >= 3.6 
- GPU, Use NVIDIA TITAN Xp with 12G RAM  
- Chinese scripts could be found in [Google Drive](https://drive.google.com/file/d/1TxY_Z_SdvIW-7BnXmjDE3gpfpVEzu22_/view?usp=sharing). Please refer to the [description](./glyce/fonts/README.md) and download scripts files to `glyce/glyce/fonts/`. 


**NOTE**: Some experimental results are obtained by training on multi-GPU machines. May use DIFFERENT PyTorch versions refer to previous open-source SOTA models. Experiment environment for Glyce-BERT is Python 3.6 and PyTorch 1.10.


## Installation

```bash 
# Clone glyce 
git clone git@github.com:ShannonAI/glyce.git
cd glyce 
python3.6 setup.py develop 

# Install package dependency 
pip install -r requirements.txt
```


## Quick start of Glyce 

### Usage of Glyce Char/Word Embedding 

```python 

import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 


from glyce import GlyceConfig 
from glyce import CharGlyceEmbedding, WordGlyceEmbedding 


# load glyce hyper-params
glyce_config = GlyceConfig()


# if input ids is char-level
glyce_embedding = CharGlyceEmbedding(glyce_config)
# elif input ids is word-level 
glyce_embedding = WordGlyceEmbedding(glyce_config)


# forward input_ids into embedding layer 
glyce_embedding, glyph_loss = glyce_embedding(input_ids)


# utilize image classifiation loss as the auxiliary loss 
glyph_decay = 0.1 
glyph_ratio = 0.01 # the proportion of image classification loss to total loss 
current_epoch = 3 


# compute loss 
loss_fct = CrossEntropyLoss()
task_loss = loss_fct(logits, labels)
total_loss = task_loss * (1 - glyce_ratio) + glyph_ratio * glyph_decay ** (current_epoch+1) * glyph_loss 
```



### Usage of Glyce-BERT

#### 1. Preparation 

- Download and unzip [`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) pretrained model. 

- Install PyTorch pretrained bert by pip as follows: 

```bash 
pip install pytorch-pretrained-bert
```
- Convert TF checkpoint into PyTorch

```shell 
export BERT_BASE_DIR=/path/to/bert/chinese_L-12_H-768_A-12

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
$BERT_BASE_DIR/bert_model.ckpt \
$BERT_BASE_DIR/bert_config.json \
$BERT_BASE_DIR/pytorch_model.bin
```

#### 2. Train/Dev/Test DataFormat

- Sequence Labeling Task 

```text
模型的输入: [CLS] 我 爱 北 京 天 安 门
模型的输出: ‘我爱北京天安门' 对应的命名实体标签 O  O  B-LOC M-LOC M-LOC M-LOC E-LOC 

Model Input: [CLS] I like Bei Jing Tian An Men
Model Output: labels corresponds to 'I like BeiJing Tian An men' are O  O  B-LOC M-LOC M-LOC M-LOC E-LOC 
```

- Sentence Pair Classification 


```text 
模型的输入: [CLS] 我 爱 北 京 天 安 门 [SEP] 北 京 欢 迎 您
模型的输出: [CLS] 位置对应的输出,假如是语意匹配任务则为 0,代表两句话的语意不相同

Model Input: [CLS] I like Bei Jing Tian An Men [SEP] Bei Jing Wel ##come you 
Model Output: the output in [CLS] position. 0 if task sign is semantic matching. It means that the semantic of two sentences are different. 
```

- Single Sentence Classification 

```text
模型的输入: [CLS] 我 爱 北 京 天 安 门
模型的输出: [CLS] 位置对应的输出,假如是情感分类任务则为1,代表情感为积极

Model Input: [CLS] I like Bei Jing Tian An Men. 
Model Output: the output in [CLS] position. 1 if task is sentiment analysis. It means the sentiment polarity is positive. 
```

#### 3. Start Train and Evaluate Glyce-BERT 

* `scritps/*_bert.sh` are the commands we used to finetune BERT.
* `scripts/*_glyce_bert.sh` are the commands we used to obtained the results of Glyce-BERT. 
* `scripts/ctb5_binaffine.sh` is the command that we used to reimplement PREVIOUS SOTA result on CTB5 for dependency parsing. 
* `scripts/ctb5_glyce_binaffine.sh` is the command that we used to obtain the SOTA result on CTB5 for dependency parsing. 

For example, training command of Glyce-BERT for sentence pair dataset BQ is included in `scripts/glyce_bert/bq_glyce_bert.sh`. 
Start train and evaluate Glyce-BERT on BQ by,

```bash 
bash scripts/glyce_bert/bq_glyce_bert.sh
```

Notes:  
- `repo_path` refer to work directory of glyce.
- `config_path` refer to the path of configuration file.
- `bert_model` refer to the the directory of the pre-trained Chinese BERT model.  
- `task_name` refer to the task signiature. 
- `data_dir` and `output_dir` refer to the directories of the "raw data" and "intermediate checkpoints" respectively. 


## Folder Description

## Implementation 

Glyce toolkit provides implementations of previous SOTA models incorporated with Glyce embeddings. 

- [Glyce: Glyph-vectors for Chinese Character Representations](https://arxiv.org/abs/1901.10125). 
Refer to [./glyce/models/glyce_bert](./glyce/models/glyce_bert)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). 
Refer to [./glyce/models/bert](./glyce/models/bert)
- [Chinese NER Using Lattice LSTM](https://arxiv.org/abs/1805.02023). 
Refer to [./glyce/models/latticeLSTM](./glyce/models/latticeLSTM)
- [Character-based Joint Segmentation and POS Tagging for Chinese using Bidirectional RNN-CRF](https://arxiv.org/abs/1704.01314). 
Refer to [./glyce/models/latticeLSTM/model](./glyce/models/latticeLSTM)
- [Syntax for Semantic Role Labeling, To Be, Or Not To Be](https://www.aclweb.org/anthology/P18-1192). 
Refer to [./glyce/models/srl](./glyce/models/srl)
- [Bilateral Multi-Perspective Matching for Natural Language Sentences](https://arxiv.org/abs/1702.03814). 
Refer to [./glyce/models/bimpm](./glyce/models/bimpm)
- [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/abs/1611.01734). 
Refer to [./glyce/models/biaffine](./glyce/models/biaffine)


## Download Task Data 

[Download Task Data](./docs/dataset_download.md)


## Welcome Contributions to Glyce Open Source Project 

We actively welcome researchers and practitioners to contribute to Glyce open source project. Please read this [Guide](https://help.github.com/en/articles/creating-a-pull-request) and submit your **Pull Request**.


## Acknowledgement      

 [Acknowledgement](./docs/acknowledge.md). Vanilla Glyce is developed based on the previous SOTA model. Glyce-BERT is developed based on [PyTorch implementation by HuggingFace](https://github.com/huggingface/pytorch-pretrained-BERT). And pretrained BERT model is released by [Google's pre-trained models](https://github.com/google-research/bert).


## Contact 

Please feel free to discuss paper/code through issues or emails.


### License 
[Apache License 2.0](./LICENSE)
