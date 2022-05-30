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
from tqdm import tqdm
import re

from utils.misc import init_vocab
from transformers import *



def encode_dataset(dataset, vocab, tokenizer, test = False):    #dataset 如果是  trainset 和val_set 是个list，list的元素是dict ,每个dict 有5个 key['question', 'choices', 'program', 'sparql', 'answer']，但是 testset里面只有['question', 'choices'])
    questions = []
    sparqls = []
    for item in tqdm(dataset):
        question = item['question']
        questions.append(question)
        if not test:       #test 数据集没有spartql
            sparql = item['sparql']
            sparqls.append(sparql)
    sequences = questions + sparqls
    encoded_inputs = tokenizer(sequences, padding = True)
    print(encoded_inputs.keys())
    print(encoded_inputs['input_ids'][0])
    print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    print(tokenizer.decode(encoded_inputs['input_ids'][-1]))
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
        _ = [vocab['answer_token_to_idx'][w] for w in item['choices']]
        choices.append(_)
        if not test:
            sparql = item['sparql']
            sparqls.append(sparql)
            answers.append(vocab['answer_token_to_idx'].get(item['answer']))

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
    return source_ids, source_mask, target_ids, choices, answers



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
        outputs = encode_dataset(dataset, vocab, tokenizer, name=='test')
        assert len(outputs) == 5
        print('shape of input_ids of questions, attention_mask of questions, input_ids of sparqls, choices and answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)
if __name__ == '__main__':
    main()