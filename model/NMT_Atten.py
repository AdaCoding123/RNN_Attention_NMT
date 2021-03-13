# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class NMT_Atten(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_embed_dim, target_embed_dim, source_embeddings,\
                            target_embeddings, hidden_size, target_max_len, bos_id):
        super(NMT_Atten, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.max_len = target_max_len
        self.bos_id = bos_id
        self.encoder = EncoderRNN(source_vocab_size, source_embed_dim, source_embeddings, hidden_size)
        # self.decoder = DecoderRNN(target_vocab_size, target_embed_dim, target_embeddings, hidden_size)
        self.decoder = AttentionDecoderRNN(target_vocab_size, target_embed_dim, target_embeddings, hidden_size)

    def forward(self, source_wordid, target_wordid):
        # source_wordid: batch_size, seq_len
        # target_wordid: batch_size, seq_len (include <bos> and <eos>)
        # hn/cn: num_layers * num_directions, batch_size, hidden_size
        output, hn, cn = self.encoder(source_wordid)
        #是什么含义
        hn = torch.cat((hn[-2], hn[-1]), -1)
        cn = torch.cat((cn[-2], cn[-1]), -1)
        # print(hn.size(), cn.size())
        # hn = torch.sum(hn, 0)
        # cn = torch.sum(cn, 0)
        #将tensor的维度换位
        target_wordid = target_wordid.permute(1, 0)
        target_wordid = target_wordid[:-1]
        #不含有最后一行
        # pred: seq_len, batch_size, target_vocab_size
        #其中的0和1表示什么意思
        pred = torch.zeros((target_wordid.size(0), target_wordid.size(1), self.target_vocab_size)).float()
        #查看是否有可用GPU
        if torch.cuda.is_available():
            pred = pred.cuda()
            #自增
        for ii, nextword_id in enumerate(target_wordid):
            linear_out, hn, cn = self.decoder(hn, cn, nextword_id, output)
            pred[ii] = linear_out

        return pred.permute(1, 0, 2)

    def translate(self, source_wordid):
        # source_wordid: batch_size, seq_len
        output, hn, cn = self.encoder(source_wordid)
        hn = torch.cat((hn[-2], hn[-1]), -1)
        cn = torch.cat((cn[-2], cn[-1]), -1)
        # hn = torch.sum(hn, 0)
        # cn = torch.sum(cn, 0)
        #torch.LongTensor是64位整型
        nextword_id = torch.LongTensor([self.bos_id]).expand(source_wordid.size(0))
        pred_word = torch.zeros((self.max_len, source_wordid.size(0))).long()
        if torch.cuda.is_available():
            nextword_id = nextword_id.cuda()
            pred_word = pred_word.cuda()
        for ii in range(self.max_len):
            linear_out, hn, cn = self.decoder(hn, cn, nextword_id, output)
            # pred: batch_size
            #每行最大值
            _, pred = torch.max(linear_out, 1)
            nextword_id = pred
            pred_word[ii] = pred

        return pred_word.permute(1, 0)


# Encoder
class EncoderRNN(nn.Module):
#source_vocab_size是源词表大小，num_laye为什么等于1
#source_vocab_size是源输入词表的大小，
#target_vocab_size是目标输出词表的大小。'''
    def __init__(self, source_vocab_size, emb_dim, embedding_weights, hidden_size, num_layer=1, finetuning=True):
        super(EncoderRNN, self).__init__()
#nn.Embedding这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。
#Parameters
#hidden_size 隐层状态的维数：（每个LSTM单元或者时间步的输出的ht的维度，单元内部有权重与偏差计算）
#num_layers RNN层的个数：（在竖直方向堆叠的多个相同个数单元的层数）
#bidirectional 是否是双向RNN，默认为false'''
        self.word_embed = nn.Embedding(source_vocab_size, emb_dim)
        #torch.tensor是一个包含多个同类数据类型数据的多维矩阵。
        if isinstance(embedding_weights, torch.Tensor):
        #首先可以把nn.Parameter理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)
            self.word_embed.weight = nn.Parameter(embedding_weights, requires_grad=finetuning)
       #该模块一次构造完若干层的LSTM。
        self.lstm = nn.LSTM(emb_dim, hidden_size//2, num_layers=num_layer, bidirectional=True)
        
         #forward函数中，传入源语言src，经过embedding层将其转换为密集向量，然后将这些词嵌入传递到LSTM。
    def forward(self, wordid_input):
        # wordid_input: batch_size, seq_len
        input_ = self.word_embed(wordid_input) # input_: batch_size, seq_len, emb_dim
        input_ = input_.permute(1, 0, 2) # input_: seq_len, batch_size, emb_dim,将tensor的维度换位
        output, (hn, cn) = self.lstm(input_)
        #hn,cn是什么？
        output = output.permute(1, 0, 2)  # output: batch_size, seq_len, hidden_size
        return output, hn, cn


# Decoder
'''class DecoderRNN(nn.Module):
    #每个cell里神经元的个数就是hidden_size
    def __init__(self, target_vocab_size, emb_dim, embedding_weights, hidden_size, finetuning=True):
        super(DecoderRNN, self).__init__()
        self.word_embed = nn.Embedding(target_vocab_size, emb_dim)
        if isinstance(embedding_weights, torch.Tensor):
            self.word_embed.weight = nn.Parameter(embedding_weights, requires_grad=finetuning)
            #该模块构建LSTM中的一个Cell，nn.RNN(input_size, hidden_size）
        self.lstmcell = nn.LSTMCell(emb_dim, hidden_size)
        #？？？
        # nn.Linear（）是用于设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]，
        self.hidden2word = nn.Linear(hidden_size, target_vocab_size)
    #函数的作用   
    def forward(self, hidden, cell, nextword_id, encoder_output=None):
        # hidden/cell: batch_size, hidden_size
        # encoder_hidden = torch.sum(encoder_hidden, 0)
        # encoder_cell = torch.sum(encoder_cell, 0)
        input_ = self.word_embed(nextword_id) # batch_size, emb_dim
        # print(input_.size())
        # print(hidden.size())
        # print(cell.size())
        h1, c1 = self.lstmcell(input_, (hidden, cell))
        output = self.hidden2word(h1)
        return output, h1, c1
'''

# Attention Decoder
class AttentionDecoderRNN(nn.Module):
    def __init__(self, target_vocab_size, emb_dim, embedding_weights, hidden_size, finetuning=True):
        super(AttentionDecoderRNN, self).__init__()

        self.word_embed = nn.Embedding(target_vocab_size, emb_dim)
        if isinstance(embedding_weights, torch.Tensor):
            self.word_embed.weight = nn.Parameter(embedding_weights, requires_grad=finetuning)
       # 原因'''
        self.lstmcell = nn.LSTMCell(emb_dim+hidden_size, hidden_size)
        # self.attention = nn.Linear(2*hidden_size, 1)
        v_size = int(0.5*hidden_size)
        self.w1 = nn.Linear(hidden_size, v_size, bias=False)
        self.w2 = nn.Linear(hidden_size, v_size, bias=False)
        #为什么用tanh'''
        self.tanh = nn.Tanh()
        self.v = nn.Linear(v_size, 1, bias=False)
        self.hidden2word = nn.Linear(hidden_size, target_vocab_size)

    def attention(self, hidden, encoder_output):
        #Q:Si-1
        atten1 = self.w1(hidden).unsqueeze(1) # batch_size, 1, v_size
        #encoder_output(batch_size,seq_len,hidden_size)
        #K:h1...hT
        atten2 = self.w2(encoder_output) # batch_size, seq_len, v_size
        #计算f(Q,K),使用加性模型
        atten3 = self.tanh(atten1+atten2) # batch_size, seq_len, v_size
        atten = self.v(atten3).squeeze(-1) # batch_size, seq_len
        ##对每一行进行softmax --- dim = -1，行和为1
        atten_weight = F.softmax(atten, -1).unsqueeze(1) # batch_size, 1, seq_len
        #加权求和
        atten_encoder_hidden = torch.bmm(atten_weight, encoder_output).squeeze(1) # batch_size, hidden_size
        return atten_encoder_hidden
        #squeeze（arg）是删除第arg个维度(如果当前维度不为1，则不会进行删除)
        
    def forward(self, hidden, cell, nextword_id, encoder_output):
        # hidden/cell: batch_size, hidden_size
        # encoder_output: batch_size, seq_len, hidden_size
        input_ = self.word_embed(nextword_id) # batch_size, emb_dim
        # batch_size, seq_len, hidden_size = encoder_output.size()
        # Q = hidden.unsqueeze(1).expand(batch_size, seq_len, hidden_size)
        # X = torch.cat((encoder_output, Q), 2)
        # atten = self.attention(X).squeeze(-1) # batch_size, seq_len
        # atten_weight = F.softmax(atten, -1).unsqueeze(1) # batch_size, 1, seq_len
        # atten_encoder_hidden = torch.bmm(atten_weight, encoder_output).squeeze(1) # batch_size, hidden_size
        atten_encoder_hidden = self.attention( hidden, encoder_output)
        #torch.cat是将两个张量（tensor）拼接在一起，按列拼接
        input_ = torch.cat((input_, atten_encoder_hidden), 1)
        h1, c1 = self.lstmcell(input_, (hidden, cell))
        output = self.hidden2word(h1)
        
        return output, h1, c1
