#使用进度条
import time
from tqdm import tqdm, trange
#trange(i)是tqdm(range(i))的一种简单写法
for i in trange(100):     #进度条的使用
    time.sleep(0.05)
for i in tqdm(range(100), desc='Processing'):  #进度条前面增加describe描述
    time.sleep(0.05)
dic = ['a', 'b', 'c', 'd', 'e']
pbar = tqdm(dic)
for i in pbar:
    pbar.set_description('Processing '+i)   #在迭代的时候，展示描述，同时展示当前正在处理的元素。
    time.sleep(0.2)



#json 解析
import json
vocab = json.load(open('dataset/vocab.json'))
inputs=[]
with open('dataset/train.pt', 'rb') as f:
    for _ in range(1):
        inputs.append(pickle.load(f))
with open(fn, 'w') as f:
    json.dump(vocab, f, indent=2)


# 字典key 和value 反转 ，KQA自己写的脚本
#  vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])