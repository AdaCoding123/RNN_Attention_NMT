import torch
from torchtext import data
from torchtext.vocab import Vectors
import numpy as np
from tqdm import tqdm


def load_dataset(train_data_path, dev_data_path, test_data_path, src_wordVectors_path,\
                 tgt_wordVectors_path, train_batch_size, dev_batch_size, test_batch_size):
#lambda匿名函数，冒号之前是参数，其后是结果
    tokenize = lambda x: x.split()
    
    src_TEXT = data.Field(sequential=True, tokenize=tokenize, pad_token='<pad>',\
                                    lower=True, include_lengths=True, batch_first=True)
    tgt_TEXT = data.Field(sequential=True, tokenize=tokenize, pad_token='<pad>',\
                                    lower=True, include_lengths=True, batch_first=True)
#导入自己的数据集
    train_data = data.TabularDataset(path=train_data_path, format='tsv',
                                fields=[('srctext', src_TEXT), ('tgttext', tgt_TEXT)])
    if dev_data_path:
        dev_data = data.TabularDataset(path=dev_data_path, format='tsv',
                                     fields=[('srctext', src_TEXT), ('tgttext', tgt_TEXT)])
    if test_data_path:
        test_data = data.TabularDataset(path=test_data_path, format='tsv',
                                fields=[('srctext', src_TEXT), ('tgttext', tgt_TEXT)])
    #构建词典，不使用词向量
    if src_wordVectors_path:
        vectors = Vectors(name=src_wordVectors_path)
        src_TEXT.build_vocab(train_data, vectors=vectors)
        src_word_embeddings = src_TEXT.vocab.vectors
        print ("Vector size of source Text Vocabulary: ", src_TEXT.vocab.vectors.size())
    else:
        src_TEXT.build_vocab(train_data)
        src_word_embeddings = None
    
    if tgt_wordVectors_path:
        vectors = Vectors(name=tgt_wordVectors_path)
        tgt_TEXT.build_vocab(train_data, vectors=vectors)
        tgt_word_embeddings = tgt_TEXT.vocab.vectors
        print ("Vector size of target Text Vocabulary: ", tgt_TEXT.vocab.vectors.size())
    else:
        tgt_TEXT.build_vocab(train_data)
        tgt_word_embeddings = None

#迭代
    train_iter = data.Iterator(train_data, batch_size=train_batch_size, \
                                       train=True, sort=False, repeat=False, shuffle=True)
    dev_iter = None
    if dev_data_path:
        dev_iter = data.Iterator(dev_data, batch_size=dev_batch_size, \
                             train=False, sort=False, repeat=False, shuffle=True)
    test_iter = None
    if test_data_path:
        test_iter = data.Iterator(test_data, batch_size=test_batch_size, \
                                     train=False, sort=False, repeat=False, shuffle=False)

    src_vocab_size = len(src_TEXT.vocab)
    tgt_vocab_size = len(tgt_TEXT.vocab)
    # print(src_TEXT.vocab.itos)
    # print(len(src_TEXT.vocab.itos))
    # print(src_vocab_size)
    # exit()

    # label_dict = dict(LABEL.vocab.stoi)
    # length = len(label_dict)
    # label_dict["<START>"] = length
    # label_dict["<STOP>"] = length+1

    return src_TEXT, tgt_TEXT, src_vocab_size, tgt_vocab_size, src_word_embeddings, tgt_word_embeddings, \
           train_iter, dev_iter, test_iter

#随机划分样本数据为训练集和验证集
def train_dev_split(train_iter, ratio):
    length = len(train_iter)
    train_data = []
    dev_data = []
    train_start = 0
    train_end = int(length*ratio)
    ind = 0
    for batch in train_iter:
        if ind < train_end:
            train_data.append(batch)
        else:
            dev_data.append(batch)
        ind += 1
    return train_data, dev_data


# if __name__ == '__main__':
#     # pass
