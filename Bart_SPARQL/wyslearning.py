import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
import json
from tqdm import tqdm
from datetime import date
from utils.misc import MetricLogger, seed_everything, ProgressBar
from utils.load_kb import DataForSPARQL
from .data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from .sparql_engine import get_sparql_answer
import torch.optim as optim
import logging
import time
from utils.misc import invert_dict
from utils.lr_scheduler import get_linear_schedule_with_warmup
import re
import json
import pickle
from itertools import chain
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query


vocab = json.load(open('dataset/vocab.json'))
vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
#print(print(list(vocab['answer_idx_to_token'].items())[:10]) )
print(len(vocab))

inputs = []
with open('dataset/train.pt', 'rb') as f:
    for _ in range(1):
        inputs.append(pickle.load(f))
print(type(inputs[0]))   #<class 'numpy.ndarray'>

# print(vocab['answer_token_to_idx'])
# print(vocab['answer_idx_to_token'] )

train_set = json.load(open('dataset/train.json'))
val_set = json.load(open('dataset/val.json'))
test_set = json.load(open('dataset/test.json'))

print(type(test_set[0]))  # trainset 和val_set 是个list，list的元素是dict ,每个dict 有5个 key['question', 'choices', 'program', 'sparql', 'answer']，但是 testset里面只有['question', 'choices'])
print(test_set[0].keys())  # trainset 是个list，list的元素是dict  
print(test_set[0])  # trainset 是个list，list的元素是dict  
# print("trainkey {}".format(train_set.keys()))
# print("val_setkey {}".format(val_set.keys()))
# print("test_set {}".format(test_set.keys()))
for question in chain(train_set, val_set, test_set):
    print(type(question))
    print(type(question.keys()))
    print(question.keys())
    break
    # for a in question['choices']:
    #     if not a in vocab['answer_token_to_idx']:
    #         vocab['answer_token_to_idx'][a] = len(vocab['answer_token_to_idx'])
#运行命令 python -m Bart_SPARQL.wyslearning
#切换conda环境命令  conda activate kqa_bart_sparql