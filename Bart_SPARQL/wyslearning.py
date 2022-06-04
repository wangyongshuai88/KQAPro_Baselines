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
from Bart_SPARQL.predict import train
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
# #切换conda环境命令  conda activate kqa_bart_sparql3.7
# train_set = json.load(open( 'dataset/train.json'))  
# tokenizer = BartTokenizer.from_pretrained('Bart_SPARQL/ckpt')   # 用来把文本进行切分，然后把词语或者短语 翻译成预训练模型对应的序号。
# questions = []
# sparqls = []
# iter2=0
# taget=set()
# for item in tqdm(train_set):     #使用进度条来展示
#     if item['sparql'].startswith('SELECT DISTINCT ?e') or item['sparql'].startswith('SELECT ?e') or item['sparql'].startswith('SELECT (COUNT(DISTINCT ?e)') or item['sparql'].startswith('SELECT DISTINCT ?p ') or  item['sparql'].startswith('ASK'):
#         continue
#     taget.add(item["sparql"].split()[2])
#     question = item['question']
#     questions.append(question)
#     sparql = item['sparql']
#     sparqls.append(sparql)
#     #print(f"{question}的查询语句是{sparql}")
# sequences = questions + sparqls  # 把问题和答案续成一个长list
# print(taget)




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

# d={'a':"1","b":'2',"c":"3"}
# dd={k if k!='b' else 'B':v if v!='2' else '9' for k,v in d.items() }
# print(dd)


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
#-------------------------------------------------------------------------------------------------------------------------------------------------------
import rdflib
from rdflib import URIRef, BNode, Literal, XSD
from rdflib.plugins.stores import sparqlstore
from itertools import chain
from tqdm import tqdm
import argparse

from utils.load_kb import DataForSPARQL
from utils.value_class import ValueClass
virtuoso_address = "http://127.0.0.1:8890/sparql"
virtuoso_graph_uri = 'KQAPro'

train_set = json.load(open("dataset/train.json")) 
iter1=1
expq,exps,expa=train_set[iter1]["question"],train_set[iter1]["sparql"],train_set[iter1]["answer"]
print(expq)
print(exps)
print(expa)
endpoint = virtuoso_address
store=sparqlstore.SPARQLUpdateStore(endpoint)
gs = rdflib.ConjunctiveGraph(store)
gs.open((endpoint, endpoint))

gs1 = gs.get_context(rdflib.URIRef(virtuoso_graph_uri))
#第一部分 
# res = gs1.query(exps)
# res = gs1.query('SELECT ?e WHERE { { ?e <pred:name> "Charlotte" . ?e <population> ?pv_1 . ?pv_1 <pred:unit> "1" . ?pv_1 <pred:value> "82675"^^xsd:double . } UNION { ?e <pred:name> "Chandler" . ?e <office_held_by_head_of_government> ?e_1 . ?e_1 <pred:name> "mayor" . } ?e <elevation_above_sea_level> ?pv . ?pv <pred:value> ?v .  } ORDER BY DESC( ?v) LIMIT 1')
# res = gs1.query('SELECT DISTINCT ?qpv WHERE { ?e <pred:name> "Georgia national football team" . ?e <ranking> ?pv . ?pv <pred:unit> "1" . ?pv <pred:value> "78"^^xsd:double . [ <pred:fact_h> ?e ; <pred:fact_r> <ranking> ; <pred:fact_t> ?pv ] <review_score_by> ?qpv .  }')
res = gs1.query('SELECT DISTINCT ?v,?u,(str(?v) as ?sv) WHERE {{ <nodeID://b41723> <pred:value> ?v  . <nodeID://b41723> <pred:value> ?u .  }}')

print(f"查询结果是{res}")
if(res.vars):
    print(res.vars)
    res = [[binding[v] for v in res.vars] for binding in res.bindings]
    print(res)
    print(res[0][0])
    print(res[0][0].value)
# print(type(res))
# print(res)
# 第二部分
# kb = DataForSPARQL('/home/shuidonger/KBQA/KQAPro_Baselines/dataset/kb.json')
# for iter1 in train_set:
#     sparql=iter1["sparql"]
#     tokens = sparql.split()
#     tgt = tokens[2]   #其他的难道都是 target 在sparql 的第三个吗？
#     if tgt !="?qpv" and tgt!="?pv":
#         continue
#     for i in range(len(tokens)-1, 1, -1):   #王永帅备注：倒序
#         if tokens[i]=='.' and tokens[i-1]==tgt:
#             key = tokens[i-2]
#             break
#     key = key[1:-1].replace('_', ' ')   #把下划线替换成空格   这是由于kb.json里面predict里是空格分隔。而在  trainset 以及validset 以及在virtuoso中都是用的下划线分割。
#     t = kb.key_type[key]
#     if t=='quantity':
#         print(sparql)
#         print(iter1["answer"])
#         res = gs1.query(sparql)
#         res = [[binding[v] for v in res.vars] for binding in res.bindings]
#         print(res)
#         print(res[0][0])
#         print(res[0][0].value)
#         parse_type = 'attr_{}'.format(t)
#-------------------------------------------------------------------------------------------------------------------------------------------------------
# Who is the reviewer of the Georgia national football team, which is ranked 78th?
# SELECT DISTINCT ?qpv WHERE { ?e <pred:name> "Georgia national football team" . ?e <ranking> ?pv . ?pv <pred:unit> "1" . ?pv <pred:value> "78"^^xsd:double . [ <pred:fact_h> ?e ; <pred:fact_r> <ranking> ; <pred:fact_t> ?pv ] <review_score_by> ?qpv .  }
# 得出结果：nodeID://b229392
# FIFA
#SELECT DISTINCT ?v WHERE {{ <Q49272> <pred:name> ?v .  }}

#  select ?e where ?e <pred:name> "Dominican Republic"
# # 找答案啊：
#b229392
# iterc1=1
# for k,i in entities.items():
#     if k!="Q786":
#     #   #  print(k)
#          continue
#     print('******************************')
#     print("key:{}".format(k) ,end="")
#     print("name:{}".format(i['name']),end="")
#     print(i)
#     # if len(i['instanceOf'])>0:
#     #     nextkey=i['instanceOf'][0]
#     #     findnexthop(sconcepts,nextkey)
#     iterc1=iterc1+1
#     if iterc1>=10:
#          break




#  给这个github 做个提交吧，
#  1  文档问题解决
#  2  知识库如果起不来，那么就用一个方法解决。
#  3  4.2.2 无法使用的问题。（观察是否有wrapper 的问题）


#  "/usr/local/virtuoso-opensource/var/lib/virtuoso/db/virtuoso.ini"`