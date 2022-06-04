# 该程序是对数据进行数据预处理的。
#python -m Bart_SPARQL.preprocess --input_dir ./dataset --output_dir <dir/of/processed/files> --model_name_or_path <dir/of/pretrained/BartModel>
# cp ./dataset/kb.json <dir/of/processed/files>
 # 使用该数据集可以做选择题，也可以做问答题。
import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from collections import Counter
from itertools import chain
from tqdm import tqdm    #进度条的包
import re

from utils.misc import init_vocab
from transformers import *



def encode_dataset(dataset, vocab, tokenizer, test = False):    #dataset 如果是  trainset 和val_set 是个list，list的元素是dict ,每个dict 有5个 key['question', 'choices', 'program', 'sparql', 'answer']，但是 testset里面只有['question', 'choices'])
    questions = []
    sparqls = []
    for item in tqdm(dataset):     #使用进度条来展示
        question = item['question']
        questions.append(question)
        if not test:       #test 数据集没有spartql
            sparql = item['sparql']
            sparqls.append(sparql)
    sequences = questions + sparqls  # 把问题和答案续成一个长list
    encoded_inputs = tokenizer(sequences, padding = True)   # 输入语句与输出sparql 没有对应关系的一个encoded 输入
    print(encoded_inputs.keys())    #dict_keys(['input_ids', 'attention_mask'])
    print(encoded_inputs['input_ids'][0]) #[0, 32251, 1139, 34, 10, 3842, 2688, 9, 204, 45121, 406, 34916, 3416, 1360, 8, 34, 41, 8192, 7961, 5135, 9, 6178, 398, 35534, 116,
    # 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    #   Which town has a TOID of 4000000074573917 and has an OS grid reference of SP8778?
    #<s>Which town has a TOID of 4000000074573917 and has an OS grid reference of SP8778?</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    print(tokenizer.decode(encoded_inputs['input_ids'][-1]))
    #<s>SELECT DISTINCT?qpv WHERE {?e <pred:instance_of>?c.?c <pred:name> "visual artwork".?e <official_website>?pv_1.?pv_1 <pred:value> "http://www.thesiege.com/".?e <publication_date>?pv.?pv <pred:date> "1999-01-21"^^xsd:date. [ <pred:fact_h>?e ; <pred:fact_r> <publication_date> ; <pred:fact_t>?pv ] <place_of_publication>?qpv.  }
    # </s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    # <pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
    max_seq_length = len(encoded_inputs['input_ids'][0])
    assert max_seq_length == len(encoded_inputs['input_ids'][-1])
    print(max_seq_length)
    questions = []
    sparqls = []
    choices = []
    answers = []
    for item in tqdm(dataset):
        question = item['question']
        questions.append(question)
        _ = [vocab['answer_token_to_idx'][w] for w in item['choices']]  #把所有的选项序号都留下来
        choices.append(_)
        if not test:
            sparql = item['sparql']
            sparqls.append(sparql)
            answers.append(vocab['answer_token_to_idx'].get(item['answer'])) #得到答案

    input_ids = tokenizer.batch_encode_plus(questions, max_length = max_seq_length, pad_to_max_length = True, truncation = True)
    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)
    if not test:
        target_ids = tokenizer.batch_encode_plus(sparqls, max_length = max_seq_length, pad_to_max_length = True, truncation = True)
        target_ids = np.array(target_ids['input_ids'], dtype = np.int32)
    else:
        target_ids = np.array([], dtype = np.int32)
    choices = np.array(choices, dtype = np.int32)
    answers = np.array(answers, dtype = np.int32)
    return source_ids, source_mask, target_ids, choices, answers  #source_ids 已经encode, source_mask 对应的mask, target_ids encode了的sparql, choices, answers  ，但是问题是为什么target 没有mask



def main():
    #python -m Bart_SPARQL.preprocess --input_dir ./dataset --output_dir <dir/of/processed/files> --model_name_or_path <dir/of/pretrained/BartModel>
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    args = parser.parse_args()

    print('Build kb vocabulary')
    vocab = {
        'answer_token_to_idx': {}
    }
    print('Load questions')
    train_set = json.load(open(os.path.join(args.input_dir, 'train.json')))   # trainset 和val_set 是个list，list的元素是dict ,每个dict 有5个 key['question', 'choices', 'program', 'sparql', 'answer']，但是 testset里面只有['question', 'choices'])
    val_set = json.load(open(os.path.join(args.input_dir, 'val.json')))
    # 王永帅增加下
    train_set=train_set[10:30]     # 王永帅增加：为了减少运行负担 
    val_set=val_set[10:30]    # 王永帅增加：为了减少运行负担 
    # 王永帅增加↑
    test_set = json.load(open(os.path.join(args.input_dir, 'test.json')))
    for question in chain(train_set, val_set, test_set):           
        for a in question['choices']:
            if not a in vocab['answer_token_to_idx']:   #查找选项的答案是否在 词典库中。 如果不在，则补充词典库。这里还是数据预处理。
                vocab['answer_token_to_idx'][a] = len(vocab['answer_token_to_idx'])

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    fn = os.path.join(args.output_dir, 'vocab.json')
    print('Dump vocab to {}'.format(fn))
    with open(fn, 'w') as f:
        json.dump(vocab, f, indent=2)
    for k in vocab:
        print('{}:{}'.format(k, len(vocab[k])))
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)   # 用来把文本进行切分，然后把词语或者短语 翻译成预训练模型对应的序号。
    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(dataset, vocab, tokenizer, name=='test')   #source_ids 已经encode, source_mask 对应的mask, target_ids encode了的sparql, choices, answers  ，但是问题是为什么target 没有mask
        assert len(outputs) == 5
        print('shape of input_ids of questions, attention_mask of questions, input_ids of sparqls, choices and answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)  #把python 对象持久化存储。
if __name__ == '__main__':
    main()