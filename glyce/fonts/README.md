## Directory Description

`fonts` contains Chinese scripts and writing styles files used in Glyce.

Chinese | English | Time Period
----------- | ------------- | ------------------------------
金文 | Bronzeware script | Shang dynasty and Zhou dynasty (2000 BC – 300 BC) 
隶书 |  Clerical script | Han dynasty (200BC-200AD)  
篆书 | Seal script | Han dynasty and Wei-Jin period (100BC - 420 AD) 
魏碑 | Tablet script | Northern and Southern dynasties 420AD - 588AD 
繁体中文 | Traditional Chinese |  600AD - 1950AD (mainland China). Still currently used in HongKong and Taiwan  
简体中文 (宋体) | Simplified Chinese - Song |  1950-now 
简体中文 (仿宋体) | Simplified Chinese - FangSong | 1950-now 
草书 | Cursive script | Jin dynasty to now  
楷书 | Regular script | Three Kingdoms period to now


## Download Chinese Scripts 

- Please Download Chinese Scripts from [Google Drive](https://drive.google.com/file/d/1TxY_Z_SdvIW-7BnXmjDE3gpfpVEzu22_/view?usp=sharing). 

- The expected directory structure is as follows:

```markdown
fonts/
    dictionary.json 
    pinyin_vocab.json 
    wubi_vocab.json 
    cjk/ 
    regular_script/
    seal_script/ 
    cursive_script/ 
    bronzeware_script/ 
    clerical_script/
    tablet_script/ 
```

**NOTE** : Please refer to `./glyce/utils/render.py` if you want to change the path to Chinese scripts files.









