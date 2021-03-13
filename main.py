# -*- coding: utf-8 -*-
import time
import torch
import torch.nn.functional as F
import numpy as np
import logging
import argparse
from model import NMT_Atten
from utils import data_preprocess, evaluate, config
import copy
from tqdm import tqdm
import jieba

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

#梯度下降
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
#填充计算
def sequence_mask(lens, max_len):
    batch_size = lens.size(0)

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp # batch_size*max_len

    return mask

def train_model(model, train_iter, dev_iter, epoch, lr, loss_func, eos_id, index2tgtword):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    all_loss = 0.0
    model.train()
    ind = 0.0
    for index, batch in enumerate(train_iter):
        src_text = batch.srctext[0]
        src_origin_lens = batch.srctext[1]

        tgt_text = batch.tgttext[0]
        tgt_origin_lens = batch.tgttext[1]
        
        # batch_size = text.size()[0]
        # target = batch.label[0]
        
        if torch.cuda.is_available():
            src_text = src_text.cuda()
            tgt_text = tgt_text.cuda()
            src_origin_lens = src_origin_lens.cuda()
            tgt_origin_lens = tgt_origin_lens.cuda()
        
        #梯度置零   
        optimizer.zero_grad()
        # pred: batch_size, seq_len, target_vocab_size
        #？？？含义
        pred = model(src_text, tgt_text)
        batch_size, seq_len, target_vocab_size = pred.size()
        #取所有行，第一列到最后一列
        tgt_text_ = tgt_text[:, 1:].contiguous()
        #进行reshape，
        tgt = tgt_text_.view((-1,))
        pred = pred.contiguous().view(-1, target_vocab_size)
        loss = loss_func(pred, tgt).view(tgt_text_.size())
        mask = sequence_mask(tgt_origin_lens-1, tgt_text_.size(1))
        loss_ = (loss*mask).sum() / (mask>0).sum()

        loss_.backward()
        # clip_gradient(model, 1e-1)
        optimizer.step()

        if index % 10 == 0:
            logger.info('Epoch:%d, Idx:%d, Training Loss:%.4f', epoch, index, loss_.item())
            # dev_iter_ = copy.deepcopy(dev_iter)
            # p, r, f1, eval_loss = eval_model(model, dev_iter, id_label)
        all_loss += loss_.item()
        ind += 1

    eval_loss, score = 0.0, 0.0
    eval_loss, score = eval_model(model, dev_iter, loss_func, eos_id, index2tgtword)
    # return all_loss/ind
    return all_loss/ind, eval_loss, score

def eval_model(model, val_iter, loss_func, eos_id, index2tgtword):
    eval_loss = 0.0
    ind = 0.0
    score = 0.0
    model.eval()
    #不track梯度
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_iter)):
            src_text = batch.srctext[0]
            src_origin_lens = batch.srctext[1]

            tgt_text = batch.tgttext[0]
            tgt_origin_lens = batch.tgttext[1]
            
            if torch.cuda.is_available():
                src_text = src_text.cuda()
                tgt_text = tgt_text.cuda()
                src_origin_lens = src_origin_lens.cuda()
                tgt_origin_lens = tgt_origin_lens.cuda()
            pred = model(src_text, tgt_text)
            batch_size, seq_len, target_vocab_size = pred.size()
            tgt_text_ = tgt_text[:, 1:].contiguous()
            tgt = tgt_text_.view((-1,))
            pred = pred.contiguous().view(-1, target_vocab_size)
            loss = loss_func(pred, tgt).view(tgt_text_.size())
            mask = sequence_mask(tgt_origin_lens-1, tgt_text_.size(1))
            loss_ = (loss*mask).sum() / (mask>0).sum()
            #？？？
            eval_loss += loss_.item()

            pred_id = model.translate(src_text) # batch_size, tgt_max_len
            score += evaluate.bleu(tgt_text[:, 1:], pred_id, tgt_origin_lens-1, eos_id)
            ind += 1

    return eval_loss/ind, score/ind

def inference(model, test_id, eos_id, index2tgtword):
    model.eval()
    with torch.no_grad():
        test_ins = np.array(test_id)
        src_text = torch.from_numpy(test_ins).long()
        if torch.cuda.is_available():
            src_text = src_text.cuda()
        pred_id = model.translate(src_text) # batch_size, tgt_max_len
        for item in pred_id:
            sen = []
            for ii in item:
                if ii == eos_id:
                    break
                sen.append(index2tgtword[ii])
            print(' '.join(sen))


def main():
    args = config.config()

    if not args.train_data_path:
        logger.info("please input train dataset path")
        exit()
    # if not (args.dev_data_path or args.test_data_path):
    #     logger.info("please input dev or test dataset path")
    #     exit()

    all_ = data_preprocess.load_dataset(args.train_data_path, args.dev_data_path, args.test_data_path, \
                     args.src_embedding_path, args.tgt_embedding_path, args.train_batch_size, \
                                                         args.dev_batch_size, args.test_batch_size)
    src_TEXT, tgt_TEXT, src_vocab_size, tgt_vocab_size, src_word_embeddings, tgt_word_embeddings, \
           train_iter, dev_iter, test_iter = all_

    bos_id = dict(tgt_TEXT.vocab.stoi)['<bos>']
    eos_id = dict(tgt_TEXT.vocab.stoi)['<eos>']
    index2tgtword = tgt_TEXT.vocab.itos

    model = NMT_Atten(src_vocab_size, tgt_vocab_size, args.src_embedding_dim, args.tgt_embedding_dim, \
              src_word_embeddings, tgt_word_embeddings, args.hidden_size, args.tgt_max_len, bos_id)
    
    if torch.cuda.is_available():
        model = model.cuda()

    train_data, dev_data = data_preprocess.train_dev_split(train_iter, 0.9)
    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model,map_location='cpu'))
        while True:
            test_sent = input("Input source sentence (q exit) >>>>")
            if test_sent.lower()=='q':
                break
            #分词
            sent = ' '.join(jieba.cut(test_sent, cut_all=False))
            #print(sent)
            test_sent = src_TEXT.preprocess(sent)
            #print(test_sent)
            test_idx = [[src_TEXT.vocab.stoi[x] for x in test_sent]]
            #print(test_idx)
            inference(model, test_idx, eos_id, index2tgtword)
        return 
    
    best_score = 0.0
    for epoch in range(args.epoch):
        train_loss, eval_loss, eval_score = train_model(model, train_data, dev_data, epoch,\
                                                               args.lr, loss_func, eos_id, index2tgtword)
        
        logger.info('Epoch:%d, Training Loss:%.4f', epoch, train_loss)
        logger.info('Epoch:%d, Eval Loss:%.4f, Eval BLEU score:%.4f', epoch, eval_loss, eval_score)
        
        if eval_score > best_score:
            best_score = eval_score
            torch.save(model.state_dict(), 'results/%d_%s_%s.pt' % (epoch, 'Model', str(best_score)))

if __name__ == "__main__":
    main()
