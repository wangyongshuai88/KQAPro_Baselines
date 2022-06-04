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
import pickle
import json
vocab = json.load(open('dataset/vocab.json'))
inputs=[]
with open('dataset/train.pt', 'rb') as f:
    for _ in range(1):
        inputs.append(pickle.load(f))
fn = 'vocab.json'
with open(fn, 'w') as f:
    json.dump(vocab, f, indent=2)


# 字典key 和value 反转 ，KQA自己写的脚本
#  vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])




# 持久化存储对象
with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
    for o in outputs:
        print(o.shape)
        pickle.dump(o, f)  #把python 对象持久化存储。

#读取持久化的对象存储
with open(question_pt, 'rb') as f:
    for _ in range(5):
        inputs.append(pickle.load(f)) #   print(type(inputs[0]))   得到 <class 'numpy.ndarray'>


# 记录日志：
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.predict.log'.format(time_)))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
# args display  首先把参数都打印到日志里去。
for k, v in vars(args).items():
    logging.info(k+':'+str(v))


#迭代器  yield 关键字可以用来写迭代器
def foo(num):
    print("starting...")
    while num<10:
        num=num+1
        yield num
for n in foo(0):
    print(n)


def foo():
    print("starting...")
    while True:
        res = yield 4   #单步调试的中断点。
        print("res:",res)
g = foo()
print(next(g))
print("*"*20)
print(next(g))     # 会从单步调试终端店继续运行。直到下一个断点。
print(g.send(7))   # 会把 7传给单步调试的中断点，


# collate_fn 会被这个样调用，就是训练的时候，sample一个batch 出来以后，，通过这个函数在进行一次处理。
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])


# 字符串前加 r 可以避免转义
# 字符串前加 r 可以在大括号里写变量


# 如果查出来是node类型，那么需要用这个语句再查一遍。SELECT DISTINCT ?v WHERE { <nodeID://b229392> <pred:value> ?v .  }
# 此类任务，每次需要2个sparql 来进行得到答案，第一个sparql得到一个节点，然后第二个sparql 根据节点得到最终的值和单位