# -*- coding: utf-8 -*-
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

def bleu(origin_id, pred_id, origin_len, eosid):
    # origin_id/pred_id: batch_size, seq_len
    # len: batch_size
    if torch.cuda.is_available():
        origin_id = origin_id.cpu()
        pred_id = pred_id.cpu()
    origin_id = origin_id.numpy().tolist()
    pred_id = pred_id.numpy().tolist()

    smooth = SmoothingFunction()  # smooth function object
    all_score = 0.0
    for origin, pred, len_ in zip(origin_id, pred_id, origin_len):
        reference = [origin[:len_]]
        try:
            end_index = pred.index(eosid)
        except:
            end_index = len(pred)
        candidate = pred[:end_index+1]
        score = sentence_bleu(reference, candidate, weights=(0.6, 0.4, 0, 0), smoothing_function=smooth.method1)

        all_score += score

    return all_score/len(origin_id)