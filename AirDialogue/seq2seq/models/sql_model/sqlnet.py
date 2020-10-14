import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules.sqlnet_condition_predict import SQLNetCondPredictor


class SQLNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,
            gpu=False, use_ca=True, trainable_emb=False):

        super(SQLNet, self).__init__()

        self.use_ca = use_ca
        self.trainable_emb = trainable_emb

        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        #Word embedding
        self.agg_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, our_model=True, trainable=trainable_emb)
        self.sel_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, our_model=True, trainable=trainable_emb)
        self.cond_embed_layer = WordEmbedding(word_emb, N_word, gpu,
                self.SQL_TOK, our_model=True, trainable=trainable_emb)
        

        #Predict number of cond
        self.cond_pred = SQLNetCondPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, use_ca, gpu)


        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()

    def forward(self, q, col, col_num, pred_entry,
            gt_where = None, gt_cond=None, reinforce=False, gt_sel=None):
        B = len(q)

        cond_score = None

        x_emb_var, x_len = self.cond_embed_layer.gen_x_batch(q, col)
        col_inp_var, col_name_len, col_len = self.cond_embed_layer.gen_col_batch(col)
        max_x_len = max(x_len)

        print('x_emb_var : ', x_emb_var.size())
        print('col_inp_var : ', col_inp_var.size())
        print('x_len : ', x_len)
        # print('q : ', len(q), 'q : ', q)
        # print('col : ', len(col), 'col : ', col)

        cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, gt_where, gt_cond, reinforce=reinforce)

        return (agg_score, sel_score, cond_score)