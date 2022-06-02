import os
from typing import ItemsView
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
# train_set = json.load(open( 'dataset/train.json'))  
# tokenizer = BartTokenizer.from_pretrained('Bart_SPARQL/ckpt')   # 用来把文本进行切分，然后把词语或者短语 翻译成预训练模型对应的序号。
# questions = []
# sparqls = []
# for item in tqdm(train_set):     #使用进度条来展示
#     question = item['question']
#     questions.append(question)
#     sparql = item['sparql']
#     sparqls.append(sparql)
# sequences = questions + sparqls  # 把问题和答案续成一个长list
# print(sequences[0])


kb = json.load(open('dataset/kb.json'))
sconcepts = kb['concepts']  #concept 是一个字典， key是一个编号，对应的内容是一个字典{name：instanceOf}
entities = kb['entities']  # entity 是concept的实例化内容， key 是一个编号，  对应内容是一个字典。 {name， instanceof，attribute，relations}  其中attribute，relations 还是一个字典。
# iterc=1
# def findnexthop( kb,key):
#     print('------------------')
#     print(key)
#     print(kb[key])
#     if len(kb[key]['instanceOf'])>0:
#         nextkey=kb[key]['instanceOf'][0]
#         findnexthop(sconcepts,nextkey)
#     else:
#          return 
#     return 
# for k,i in sconcepts.items():
#     print('******************************')
#     print(k)
#     print(i)
#     if len(i['instanceOf'])>0:
#         nextkey=i['instanceOf'][0]
#         findnexthop(sconcepts,nextkey)
#     iterc=iterc+1
#     if iterc>=10:
#          break
# iterc1=1
# def findnexthop( kb,key):
#     print('**上位词**',end="")
#     print("key:{}".format(key) ,end="")
#     print("name:{}".format(kb[key]['name']))
#     if len(kb[key]['instanceOf'])>0:
#         nextkey=kb[key]['instanceOf'][0]
#         findnexthop(kb,nextkey)
#     else:
#          return 
#     return 
# for k,i in entities.items():
#     print('******************************')
#     print("key:{}".format(k) ,end="")
#     print("name:{}".format(i['name']),end="")
#     if len(i['instanceOf'])>0:
#         nextkey=i['instanceOf'][0]
#         findnexthop(sconcepts,nextkey)
#     iterc1=iterc1+1
#     if iterc1>=10:
#          break

# # and i["name"]!= 'United States of America'
# iterc2=1
# for k,i in entities.items():
#     iterc2=iterc2+1
#     if iterc2<=1 :
#       continue
#     print(k,end="")
#     print(i['name'])
#     print(type(i['attributes']))

#     for ctt1 in i['attributes']:
#         print(ctt1)
#         print(type(ctt1))
#         print('-----------------')
#     print("000000000000000000000000000000000000")
#     for ctt2 in i['relations']:
#         print(ctt2,end="对象名字是")
#         if ctt2['object'] in entities.keys():
#             print(entities[ctt2['object']]["name"])
#         elif ctt2['object'] in sconcepts.keys():
#             print("一个概念{}".format(sconcepts[ctt2['object']]["name"]))

#     if iterc2>=2:
#          break

d={'a':"1","b":'2',"c":"3"}
dd={k if k!='b' else 'B':v if v!='2' else '9' for k,v in d.items() }
print(dd)


# and i["name"]!= 'United States of America'
# iterc2=1
# for k,i in entities.items():
#     iterc2=iterc2+1
#     if i["name"]!= 'United States of America' :
#       continue
#     print(k,end="")
#     print(i['name'])
#     print(type(i['attributes']))

#     for ctt1 in i['attributes']:
#         print(ctt1)
#         print(type(ctt1))
#         print('-----------------')
#     print("000000000000000000000000000000000000")
#     for ctt2 in i['relations']:
#         print(ctt2,end="对象名字是")
#         if ctt2['object'] in entities.keys():
#             print(entities[ctt2['object']]["name"])
#         elif ctt2['object'] in sconcepts.keys():
#             print("一个概念{}".format(sconcepts[ctt2['object']]["name"]))

#     if iterc2>=2:
#          break


        
# print(type(entities['Q786']['attributes'][0]))
# print(entities['Q786']['name'])
# for ka,va in entities['Q786']['attributes'][0].items():
#     print(ka,end='')
#     print(va)



# encoded_inputs = tokenizer(sequences, padding = True)
# print(encoded_inputs.keys())
# print(encoded_inputs['input_ids'][0])
# print(tokenizer.decode(encoded_inputs['input_ids'][0]))
# print(tokenizer.decode(encoded_inputs['input_ids'][-1]))
# max_seq_length = len(encoded_inputs['input_ids'][0])
# assert max_seq_length == len(encoded_inputs['input_ids'][-1])


# Q7270
# <class 'str'>
# {'name': 'republic', 'instanceOf': ['Q7174']}
# <class 'dict'>
# Q130232
# <class 'str'>
# {'name': 'drama film', 'instanceOf': ['Q21010853']}
# <class 'dict'>
# Q280658
# <class 'str'>
# {'name': 'forward', 'instanceOf': ['Q12737077']}
# <class 'dict'>
# Q8355
# <class 'str'>
# {'name': 'violin', 'instanceOf': ['Q192096']}
# <class 'dict'>