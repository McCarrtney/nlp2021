# nlp2021
Final project of DL4NLP 2021 ZJU


### preprocess

#### image

Extract object-detection features of each image using fast r-cnn model

The tidy process Refers to **def process_image_feature**

Zipped File contains features of:

train_set: 32077 images

val_set: 15682 images

test_set: 15718 images

feature shape: (2048, 36)

```bash
# to download the Zip file
# visit https://pan.zju.edu.cn/share/c7fb4d569d8efd00b55014ed33
# unzip and use h5py to read
```

#### text

Process the question and answer text Refers to **def process_vocab** and **def encode_question** and **def encode_answer**

The vocab file will be saved at `./vocab/*.json`

#### mindspore dataset

Generate mindspore dataset and Save to Mindrecord file Refers to **def gen_mindspore_dataset**

A single sample format will be:

```python
{
    'question_id': 393226002, 
    'image_id': 393226, 
    'question': array([3, 14, 1, 113, 7, 1, 68, 1192, 4877, 4877, 4877,4877, 4877, 4877, 4877, 4877, 4877, 4877, 4877]), 
    'answer': [489, 489, 489, 489, 489, 489, 489, 489, 489, 489], 
    'answer_counter': {489: 10}, 
    'answer_label': 489
}
```

