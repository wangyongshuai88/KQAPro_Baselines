import json
import pickle
import torch
from utils.misc import invert_dict

def load_vocab(path):
    vocab = json.load(open(path))
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])   #invert_dict 用来转换字典的键值。 vocab 是一个字典里面嵌套一个字典。vocab['answer_token_to_idx'] 和vocab['answer_idx_to_token']是两个字典。
    return vocab

def collate(batch):
    batch = list(zip(*batch))   # 把 没一个例子的五个元素在一起的   转换成一类在一起：5列分别代表不同含义的集合
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    choices = torch.stack(batch[2])
    if batch[-1][0] is None:
        target_ids, answer = None, None
    else:
        target_ids = torch.stack(batch[3])
        answer = torch.cat(batch[4])
    return source_ids, source_mask, choices, target_ids, answer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids, self.choices, self.answers = inputs #self.choices, self.answers 是文本
        self.is_test = len(self.answers)==0   # 判断是否是测试集


    def __getitem__(self, index):    # 需要实现一个利用 index 取数的getitem
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        choices = torch.LongTensor(self.choices[index])
        if self.is_test:
            target_ids = None
            answer = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
            answer = torch.LongTensor([self.answers[index]])
        return source_ids, source_mask, choices, target_ids, answer


    def __len__(self):     # 并且需要一个总长度。
        return len(self.source_ids)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False):  # question pt是在数据预处理中，使用bart的tokentizer 处理好的对象存储  含 #source_ids 已经encode, source_mask 对应的mask, target_ids encode了的sparql, choices, answers  ，但是问题是为什么target 没有mask
        vocab = load_vocab(vocab_json) #vocab['answer_token_to_idx'] 和vocab['answer_idx_to_token']是两个字典。
        if training:
            print('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))
        
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(5):
                inputs.append(pickle.load(f)) #   print(type(inputs[0]))   得到 <class 'numpy.ndarray'>
        dataset = Dataset(inputs)    #inputs  含5个list  #source_ids 已经encode, source_mask 对应的mask, target_ids encode了的sparql, choices, answers  

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,  #只有训练的时候才shaffle
            collate_fn=collate, 
            )
        self.vocab = vocab