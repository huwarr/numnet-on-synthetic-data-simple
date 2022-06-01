# NumNet+ pretraining on GenBERT data

A more simple scheme for pretraining NumNet+ on synthetic numerical and textual data.

Training scheme consists of the following steps:

1. Pre-trainiing NumNet+ (without reasoning module) on synthetic numerical data.
2. Finetuning result of (1) on synthetic textual data.
3. Finetuning result of (2) or (1) on DROP.

## Sources

Data: [[CODE]](https://github.com/ag1988/injecting_numeracy), [[PDF]](https://arxiv.org/pdf/2004.04487.pdf)

NumNet+: [[CODE]](https://github.com/llamazing/numnet_plus)

## Requirements

`pip install -r requirements.txt`

## Data and pretrained RoBERTa

- Download synthetic data, already converted to DROP format.

  `mkdir synthetic_data && cd synthetic_data`
  
  Numerical:
  
  `gdown https://drive.google.com/uc?id=1WoCuawj3F1RRHG9RJ0Pfow597ASgsTf5`
  
  `gdown https://drive.google.com/uc?id=1juJczB0mQorhKOpfvE6z0_hFxP44jkUd`
  
  Textual:
  
  `gdown https://drive.google.com/uc?id=1TXZv_za1I_zC3LZ4bg2A8IJmOLpqd7xo`
  
  `gdown https://drive.google.com/uc?id=1p3OeXhpmbdrhba4P_onWveUVlPinyTig`
    
- Download DROP dataset.

  `wget -O drop_dataset.zip https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip`
   
  `unzip drop_dataset.zip`

- Download RoBERTa.

  `mkdir roberta.large && cd roberta.large`
  
  `wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin`
  
  `wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json`
  
  `wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json`
  
  `wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt`
  
  Modify **config.json** from `"output_hidden_states": false` to `"output_hidden_states": true`.

## Training

### Step 1

- Tag based multi-span extraction (NumNet+ v2):

  `sh train-ND.sh 345 1e-5 1e-5 5e-5 0.01 tag_mspan synthetic_data`
 
- Simple multi-span extraction (NumNet+):

  `sh train-ND.sh 345 1e-5 1e-5 5e-5 0.01 mspan synthetic_data`


### Step 2

**!NB**: The type of the model (i.e. NumNet+ or NumNet+v2) should be the same, as in step 1.

- Tag based multi-span extraction (NumNet+ v2):

  `sh train-TD.sh 345 1e-5 1e-5 5e-5 0.01 tag_mspan synthetic_data`
 
- Simple multi-span extraction (NumNet+):

  `sh train-TD.sh 345 1e-5 1e-5 5e-5 0.01 mspan synthetic_data`
  
### Step 3

**!NB**: The type of the model (i.e. NumNet+ or NumNet+v2) should be the same, as in steps 1, 2.

- Tag based multi-span extraction (NumNet+ v2):

  `sh finetune-DROP.sh 345 5e-4 1.5e-5 5e-5 0.01 tag_mspan drop_dataset`

- Simple multi-span extraction (NumNet+):

  `sh finetune-DROP.sh 345 5e-4 1.5e-5 5e-5 0.01 mspan drop_dataset`
