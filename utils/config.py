# -*- coding: utf-8 -*-
import logging
import argparse

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def config():
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument("--epoch", default=100, type=int,
                        help="the number of epoches needed to train")
    parser.add_argument("--lr", default=2e-3, type=float,
                        help="the learning rate")
    
    parser.add_argument("--train_data_path", default='dataset/train.tsv', type=str,
                        help="train dataset path")
    parser.add_argument("--dev_data_path", default=None, type=str,
                        help="dev dataset path")
    parser.add_argument("--test_data_path", default=None, type=str,
                        help="test dataset path")
    
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="the batch size")
    parser.add_argument("--dev_batch_size", default=64, type=int,
                        help="the batch size")
    parser.add_argument("--test_batch_size", default=64, type=int,
                        help="the batch size")
    
    parser.add_argument("--src_embedding_path", default='dataset/sgns.renmin.bigram-char', type=str,
                        help="source pre-trained word embeddings path")
    parser.add_argument("--tgt_embedding_path", default='dataset/glove.6B.300d.txt', type=str,
                        help="target pre-trained word embeddings path")
    
    parser.add_argument("--src_embedding_dim", default=300, type=int,
                        help="the source word embedding size")
    parser.add_argument("--tgt_embedding_dim", default=300, type=int,
                        help="the target word embedding size")
    
    parser.add_argument("--hidden_size", default=128, type=int,
                        help="the hidden size")
    parser.add_argument("--tgt_max_len", default=31, type=int,
                        help="the target allowed max length")
    parser.add_argument("--fine_tuning", default=True, type=bool,
                        help="whether fine-tune word embeddings")
    parser.add_argument("--early_stopping", default=15, type=int,
                        help="Tolerance for early stopping (# of epochs).")
    parser.add_argument("--load_model", default=None,
                        help="load pretrained model for testing")
    args = parser.parse_args()

    return args
