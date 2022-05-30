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
import time

#运行命令 python -m Bart_SPARQL.wyslearning
#切换conda环境命令  conda activate kqa_bart_sparql
train_set = json.load(open( 'dataset/train.json'))  
tokenizer = BartTokenizer.from_pretrained('Bart_SPARQL/ckpt')   # 用来把文本进行切分，然后把词语或者短语 翻译成预训练模型对应的序号。
questions = []
sparqls = []
for item in tqdm(train_set):     #使用进度条来展示
    question = item['question']
    questions.append(question)
    sparql = item['sparql']
    sparqls.append(sparql)
sequences = questions + sparqls  # 把问题和答案续成一个长list
print(sequences[0])
# encoded_inputs = tokenizer(sequences, padding = True)
# print(encoded_inputs.keys())
# print(encoded_inputs['input_ids'][0])
# print(tokenizer.decode(encoded_inputs['input_ids'][0]))
# print(tokenizer.decode(encoded_inputs['input_ids'][-1]))
# max_seq_length = len(encoded_inputs['input_ids'][0])
# assert max_seq_length == len(encoded_inputs['input_ids'][-1])

